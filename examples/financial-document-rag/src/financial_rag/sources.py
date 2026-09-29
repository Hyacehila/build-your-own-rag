from __future__ import annotations

import base64
import re
import time

from .common import RagError, digest
from .config import Config
from .dataset import verify_document
from .parsing import page_source
from .storage import Store
from .tokenization import Tokenizer


def render_source(store: Store, source: dict, scale: float, crop: bool = False) -> str:
    import pymupdf

    doc = store.get("documents", source["doc_version"])
    key = digest([doc["sha256"], source["page_index"], scale, source.get("bbox") if crop else None])
    dest = store.root / "page_images" / (key + ".png")
    if not dest.exists():
        verify_document(doc)
        with pymupdf.open(doc["pdf_path"]) as pdf:
            page = pdf[source["page_index"]]
            rect = pymupdf.Rect(source["bbox"]) & page.rect if crop and source.get("bbox") else page.rect
            if rect.is_empty:
                raise RagError("Empty figure bounding box; inspect parsing provenance.")
            image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
    return "data:image/png;base64," + base64.b64encode(dest.read_bytes()).decode("ascii")


class Reader:
    """The shared source service and hard evidence budget for every arm.

    Search previews have a separate bounded budget; original evidence has its own budget.
    Citation IDs are minted only after actual text/image reads, never by search.
    Text source-page coverage is conservative for multi-page table nodes: a full
    table read covers all its pages; partial text with ambiguous page offsets
    covers none until those page images or individually located nodes are read.
    """

    def __init__(self, config: Config, store: Store, index: dict):
        self.config, self.store, self.index = config, store, index
        self.tokenizer = Tokenizer(config.tokenizer)
        self.text_used = 0
        self.preview_used = 0
        self.question = ""
        self.retrieval_matches = {}
        self._read_limit = config.query.text_tokens
        self.preview_cache: dict[str, str] = {}
        self.images: dict[str, str] = {}
        self.evidence: dict[str, dict] = {}
        self.seen: dict[str, str] = {}
        self.trace: list[dict] = []
        self.allowed_versions = {store.get("parses", p)["doc_version"] for p in index["parse_ids"]}
        self.verified: set[str] = set()

    @property
    def remaining(self) -> int:
        return max(0, self.config.query.text_tokens - self.text_used)

    def preview(self, chunk: dict) -> str:
        if chunk["id"] not in self.preview_cache:
            part = self.tokenizer.prefix(
                chunk["text"],
                min(
                    self.config.query.search_preview_tokens,
                    max(0, self.config.query.preview_total_tokens - self.preview_used),
                ),
            )
            self.preview_used += self.tokenizer.count(part)
            self.preview_cache[chunk["id"]] = part
        return self.preview_cache[chunk["id"]]

    def _document(self, version: str) -> dict:
        if version not in self.allowed_versions:
            raise RagError("Source is outside this index's document versions.")
        doc = self.store.get("documents", version)
        if version not in self.verified:
            verify_document(doc)
            self.verified.add(version)
        return doc

    def _text(self, key: str, text: str, sources: list[dict], offset: int, kind: str) -> dict:
        if offset < 0 or offset > len(text):
            raise RagError("Read offset is outside the source text (Unicode character offset).")
        # Skip already-read intervals even when a later window overlaps at a
        # different character offset. A continuation consumes only new text.
        for evidence in self.evidence.values():
            if evidence.get("source_key") == key and evidence["offset"] <= offset < evidence["end_offset"]:
                return {
                    "evidence_id": evidence["id"],
                    "cached": True,
                    "next_offset": evidence["end_offset"] if evidence["end_offset"] < len(text) else None,
                }
        end_limit = min(
            [
                e["offset"]
                for e in self.evidence.values()
                if e.get("source_key") == key and e["offset"] > offset
            ]
            or [len(text)]
        )
        read_key = digest([key, offset, "text"])
        selected = self.tokenizer.prefix(
            text[offset:end_limit], min(self.remaining, max(0, self._read_limit - self.text_used))
        )
        if not selected:
            return {
                "status": "text_budget_exhausted" if text[offset:] else "empty_text",
                "next_offset": offset,
            }
        end = offset + len(selected)
        eid = f"E{len(self.evidence) + 1}"
        self.text_used += self.tokenizer.count(selected)
        pages = {(s["doc_version"], s["page_index"]) for s in sources}
        located = sources if len(pages) <= 1 or (offset == 0 and end == len(text)) else []
        evidence = {
            "id": eid,
            "type": "text",
            "kind": kind,
            "text": selected,
            "sources": sources,
            "covered_sources": located,
            "offset": offset,
            "end_offset": end,
            "truncated": end < len(text),
            "source_key": key,
        }
        self.evidence[eid] = evidence
        self.seen[read_key] = eid
        return {
            "evidence_id": eid,
            "next_offset": end if end < len(text) else None,
            "truncated": evidence["truncated"],
        }

    def _image(self, source: dict) -> dict:
        page_key = f"{source['doc_version']}:{source['page_index']}"
        if page_key in self.images:
            return {"evidence_id": self.images[page_key], "cached": True}
        if len(self.images) >= self.config.query.page_images:
            return {"status": "image_budget_exhausted"}
        eid = f"E{len(self.evidence) + 1}"
        data = render_source(self.store, source, self.config.query.image_scale)
        full_page = page_source(self._document(source["doc_version"]), source["page_index"])
        self.evidence[eid] = {
            "id": eid,
            "type": "image",
            "sources": [full_page],
            "covered_sources": [full_page],
            "data_url": data,
        }
        self.images[page_key] = eid
        return {"evidence_id": eid}

    def _node(self, node_id: str, offset: int, include_images: bool) -> list[dict]:
        node = self.store.get("nodes", node_id)
        if node["parse_id"] not in self.index["parse_ids"]:
            raise RagError("Node belongs to a different parse version.")
        if not node["sources"]:
            return [
                {
                    "node_id": node_id,
                    "status": "container",
                    "children": node["children"],
                    "parent_id": node.get("parent_id"),
                    "section_id": node.get("section_id"),
                }
            ]
        for source in node["sources"]:
            self._document(source["doc_version"])
        result = [
            {"node_id": node_id, **self._text(node_id, node["text"], node["sources"], offset, node["kind"])}
        ]
        if include_images:
            result.extend(self._image(s) for s in node["sources"])
        return result

    def _fragment(self, chunk: dict, include_images: bool, offset: int = 0) -> list[dict]:
        """Read immutable original chunk text, never a generated retrieval description.

        This keeps table headers with matching rows and uses already-localized body
        provenance without incorrectly applying raw.orig offsets to normalized text.
        """
        if chunk.get("enrichment_id"):
            raise RagError("Generated retrieval descriptions cannot become evidence.")
        for nid in chunk["node_ids"]:
            if self.store.get("nodes", nid)["parse_id"] not in self.index["parse_ids"]:
                raise RagError("Fragment belongs to a different parse version.")
        for source in chunk["sources"]:
            self._document(source["doc_version"])
        result = self._text("chunk:" + chunk["id"], chunk["text"], chunk["sources"], offset, chunk["kind"])
        eid = result.get("evidence_id")
        if eid:
            evidence = self.evidence[eid]
            evidence["chunk_id"] = chunk["id"]
            evidence["read_scope"] = "original_fragment"
            # Multi-page table rows lack exact cell-to-page spans: do not overclaim coverage.
            if chunk["kind"] == "table" and len({s["page_index"] for s in chunk["sources"]}) > 1:
                evidence["covered_sources"] = []
        output = [
            {**result, "chunk_id": chunk["id"], "node_ids": chunk["node_ids"], "continuation_scope": "chunk"}
        ]
        if include_images and chunk["kind"] in {"table", "picture"}:
            output.extend(self._image(s) for s in chunk["sources"])
        return output

    def _original_fragments(self, chunk: dict) -> list[dict]:
        if not chunk.get("enrichment_id"):
            matches = self.retrieval_matches.get(chunk["id"], [chunk["id"]])
            originals = [self.store.get("chunks", cid) for cid in matches]
            return [c for c in originals if not c.get("enrichment_id")][
                : self.config.query.asset_read_fragments
            ]
        assets = set(chunk.get("asset_ids", []))
        candidates = [
            c
            for c in self.store.chunks(self.index["id"])
            if not c.get("enrichment_id") and assets.intersection(c.get("asset_ids", []))
        ]
        terms = set(re.findall(r"\w+", self.question.lower()))
        # A summary hit is a pointer to its source. Select original rows using the actual
        # question, not benchmark labels or generated numbers; expose images as fallback.
        candidates.sort(
            key=lambda c: (-len(terms.intersection(re.findall(r"\w+", c["text"].lower()))), c["id"])
        )
        return candidates[: self.config.query.asset_read_fragments]

    def _page(self, version: str, page_index: int, offset: int, include_images: bool) -> list[dict]:
        import pymupdf

        doc = self._document(version)
        source = page_source(doc, page_index)
        with pymupdf.open(doc["pdf_path"]) as pdf:
            text = pdf[page_index].get_text("text", sort=True)
        result = [
            {
                "doc_version": version,
                "page_index": page_index,
                **self._text(f"{version}:page:{page_index}", text, [source], offset, "page"),
            }
        ]
        if include_images:
            result.append(self._image(source))
        return result

    def read(
        self,
        chunk_id: str | None = None,
        node_id: str | None = None,
        doc_version: str | None = None,
        page_index: int | None = None,
        offset: int = 0,
        include_images: bool = True,
        mode: str = "fragment",
        radius: int = 2,
        depth: int = 2,
        node_start: int = 0,
        max_tokens: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        target = {
            "chunk_id": chunk_id,
            "node_id": node_id,
            "doc_version": doc_version,
            "page_index": page_index,
            "offset": offset,
            "include_images": include_images,
            "mode": mode,
            "radius": radius,
            "depth": depth,
            "node_start": node_start,
            "max_tokens": max_tokens,
        }
        trace = {"tool": "read", "arguments": target}
        before = set(self.evidence)
        try:
            from .readback import select_nodes

            if max_tokens is not None and (
                isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
            ):
                raise RagError("max_tokens must be a positive integer.")
            self._read_limit = self.text_used + min(
                max_tokens or self.config.query.read_max_tokens, self.config.query.read_max_tokens
            )

            if not isinstance(node_start, int) or isinstance(node_start, bool) or node_start < 0:
                raise RagError("node_start must be a nonnegative integer.")
            next_node_start = None
            expanded = []
            if sum(x is not None for x in (chunk_id, node_id, doc_version)) != 1:
                raise RagError("Read exactly one chunk_id, node_id, or doc_version + page_index.")
            if chunk_id:
                chunk = self.store.get("chunks", chunk_id)
                if chunk["index_id"] != self.index["id"]:
                    raise RagError("Chunk belongs to a different index.")
                if mode == "fragment":
                    if offset and chunk.get("enrichment_id"):
                        raise RagError(
                            "Continue using the returned original chunk_id, not its generated summary."
                        )
                    originals = [chunk] if offset else self._original_fragments(chunk)
                    result = [
                        r for original in originals for r in self._fragment(original, include_images, offset)
                    ]
                    if not originals:
                        # Captionless images still have an original source, never cite the summary.
                        result = [
                            r for nid in chunk["node_ids"] for r in self._node(nid, offset, include_images)
                        ]
                elif self.index["kind"] == "flat":
                    pages = {(s["doc_version"], s["page_index"]) for s in chunk["sources"]}
                    result = [r for v, p in sorted(pages) for r in self._page(v, p, offset, include_images)]
                else:
                    expanded = select_nodes(
                        self.store,
                        self.index["parse_ids"],
                        chunk["node_ids"],
                        "window" if mode == "context" else mode,
                        radius,
                        depth,
                    )
                    if mode == "context":
                        expanded = [nid for nid in expanded if nid not in chunk["node_ids"]]
            elif node_id:
                expanded = select_nodes(
                    self.store,
                    self.index["parse_ids"],
                    [node_id],
                    "node" if mode == "fragment" else mode,
                    radius,
                    depth,
                )
            else:
                if page_index is None:
                    raise RagError("page_index is required and must be zero based.")
                result = self._page(doc_version, page_index, offset, include_images)
            if expanded or node_id or (chunk_id and self.index["kind"] != "flat" and mode != "fragment"):
                selected = expanded[node_start : node_start + 64]
                next_node_start = node_start + 64 if len(expanded) > node_start + 64 else None
                result = [r for nid in selected for r in self._node(nid, offset, include_images)]
            response = {
                "results": result,
                "expanded_nodes": expanded[node_start : node_start + 64],
                "next_node_start": next_node_start,
                "remaining_text_tokens": self.remaining,
                "remaining_page_images": self.config.query.page_images - len(self.images),
                "preview_tokens_used": self.preview_used,
            }
            trace["status"] = "ok"
        except (RagError, ValueError, TypeError, IndexError) as e:
            response = {"error": str(e)}
            trace["status"] = "error"
        trace.update(
            response=response,
            new_evidence=list(set(self.evidence) - before),
            seconds=time.perf_counter() - started,
        )
        self.trace.append(trace)
        return response

    def public_evidence(self) -> list[dict]:
        return [{k: v for k, v in e.items() if k != "data_url"} for e in self.evidence.values()]

    def content(self, only_ids: set[str] | None = None) -> list[dict]:
        parts = []
        for eid, e in self.evidence.items():
            if only_ids is not None and eid not in only_ids:
                continue
            locator = "; ".join(f"{s['doc_id']} page {s['page_number']}" for s in e["sources"])
            if e["type"] == "text":
                parts.append({"type": "text", "text": f"[{eid}] {locator}\n{e['text']}"})
            else:
                parts.extend(
                    [
                        {"type": "text", "text": f"[{eid}] Full original page: {locator}"},
                        {"type": "image_url", "image_url": {"url": e["data_url"]}},
                    ]
                )
        return parts


def validate_citations(citations: list, evidence: list[dict]) -> dict:
    available = {e["id"]: e for e in evidence if e.get("sources")}
    valid, invalid = [], []
    for citation in citations:
        if isinstance(citation, str) and citation in available:
            valid.append(citation)
        else:
            invalid.append(citation)
    return {
        "valid": valid,
        "invalid": invalid,
        "total": len(citations),
        "validity": len(valid) / len(citations) if citations else None,
        "bindings": {key: available[key]["sources"] for key in dict.fromkeys(valid)},
    }
