from __future__ import annotations

from collections import Counter

from .common import RagError, digest, versions
from .config import Config
from .dataset import dataset
from .parsing import active_parses, load_docling
from .storage import Store
from .tokenization import Tokenizer, recursive_spans


def chunk_signature(config: Config) -> dict:
    result = {
        "chunking": config.chunking.semantic_options(),
        "tokenizer": config.tokenizer.model_dump(),
        "contract": "hybrid-fact-ancestry-with-asset-boundaries-v3",
        "versions": versions("docling-core", "tiktoken"),
        "table_serializer": "MarkdownTableSerializer"
        if config.chunking.table_headers == "legacy"
        else "MultirowTableSerializer-v1",
        "merge_peers": "adjacent-body-same-section-parent-headings",
        "repeat_table_header": True,
    }
    if config.chunking.table_headers == "multirow":
        result["table_budget"] = "context-reserved-v1"
    if config.chunking.source_scope == "fragment":
        result["source_contract"] = "unique-original-fragment-v1"
    return result


def make_chunk(text: str, nodes: list[dict], kind: str, headings=None, **extra) -> dict:
    refs = list(dict.fromkeys(n["id"] for n in nodes))
    sources = list({digest(s): s for n in nodes for s in n["sources"]}.values())
    return {
        "text": text,
        "node_ids": refs,
        "sources": sources,
        "kind": kind,
        "headings": headings or [],
        **extra,
    }


def flat_chunks(nodes: list[dict], config: Config, tokenizer: Tokenizer) -> list[dict]:
    text, ranges = "", []
    for n in nodes:
        start = len(text)
        text += n["text"] + "\n\n"
        # Inserted separators are not evidence from a page. Blank physical pages
        # stay in the dataset, but must not inherit a neighbour's retrieval hit.
        if n["text"].strip():
            ranges.append((start, start + len(n["text"]), n))
    chunks = []
    for a, b in recursive_spans(text, tokenizer, config.chunking.max_tokens, config.chunking.overlap):
        if text[a:b].strip():
            hit = [n for start, end, n in ranges if start < b and end > a]
            chunks.append(make_chunk(text[a:b], hit, "text", char_start=a, char_end=b))
    return chunks


def hybrid_chunks(parsed: dict, nodes: list[dict], config: Config, tokenizer: Tokenizer) -> list[dict]:
    from docling_core.transforms.chunker import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
    from docling_core.transforms.serializer.base import BaseSerializerProvider
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
    from docling_core.types.doc import TableItem

    class Provider(BaseSerializerProvider):
        def get_serializer(self, doc):
            # Markdown table serialization supplies header/body lines to HybridChunker.
            if config.chunking.table_headers == "multirow":
                from .table_serialization import MultirowTableSerializer

                table_serializer = MultirowTableSerializer()
            else:
                table_serializer = MarkdownTableSerializer()
            return ChunkingDocSerializer(doc=doc, table_serializer=table_serializer)

    doc = load_docling(parsed)
    by_ref = {n["ref"]: n for n in nodes}

    class ContextBudgetHybridChunker(HybridChunker):
        def _split_by_doc_items(self, doc_chunk, doc_serializer):
            # The upstream heading stack follows heading levels across the entire
            # file, including independent appendix groups. Facts have explicit
            # parents, so bind ancestry BEFORE token splitting/context budgeting.
            from itertools import groupby

            def scope(item):
                node = by_ref[item.self_ref]
                return (tuple(node["headings"]), node["parent_id"], node["section_id"])

            output, start = [], 0
            for key, group in groupby(doc_chunk.meta.doc_items, scope):
                size = len(list(group))
                if start == 0 and size == len(doc_chunk.meta.doc_items):
                    part = doc_chunk.model_copy(deep=True)
                else:
                    part = self._make_chunk_from_doc_items(doc_chunk, start, start + size - 1, doc_serializer)
                part.meta.headings = list(key[0]) or None
                output.extend(super()._split_by_doc_items(part, doc_serializer))
                start += size
            return output

        def segment(self, doc_chunk, available_length, doc_serializer):
            if (
                config.chunking.table_headers == "multirow"
                and self.repeat_table_header
                and len(doc_chunk.meta.doc_items) == 1
                and isinstance(doc_chunk.meta.doc_items[0], TableItem)
            ):
                # Docling 2.91.0's table path ignores available_length and uses
                # the full tokenizer budget again. Reserve the heading context
                # computed by HybridChunker before the line-based table split.
                bounded = self.model_copy(update={"tokenizer": tokenizer.docling(available_length)})
                return HybridChunker.segment(bounded, doc_chunk, available_length, doc_serializer)
            return super().segment(doc_chunk, available_length, doc_serializer)

    chunker = ContextBudgetHybridChunker(
        tokenizer=tokenizer.docling(config.chunking.max_tokens),
        serializer_provider=Provider(),
        merge_peers=False,
        repeat_table_header=True,
    )
    output = []
    for chunk in chunker.chunk(dl_doc=doc):
        text = chunker.contextualize(chunk=chunk)
        if not text.strip():
            continue
        items = [by_ref[item.self_ref] for item in chunk.meta.doc_items]
        # Docling may emit a figure's caption as a TextItem without the PictureItem
        # in chunk metadata. Recover that original relationship so C replaces the
        # caption entrance instead of accidentally indexing it a second time.
        for item in list(items):
            if item["kind"] == "caption":
                for ref in item["related_refs"]:
                    owner = by_ref.get(ref)
                    if owner and owner["kind"] == "picture" and owner not in items:
                        items.append(owner)
        kinds = {n["kind"] for n in items}
        asset_ids = [n["id"] for n in items if n["kind"] in ("table", "picture")]
        kind = "table" if "table" in kinds else "picture" if "picture" in kinds else "text"
        if asset_ids and any(n["kind"] not in {"table", "picture", "caption", "footnote"} for n in items):
            raise RagError("HybridChunker mixed an asset with body text; refusing to drop body text in C.")
        if not any(n["sources"] for n in items):
            raise RagError(f"Chunk has no page provenance: {[n['ref'] for n in items]}")
        output.append(
            make_chunk(
                text,
                items,
                kind,
                chunk.meta.headings,
                asset_ids=asset_ids,
                body_text=chunk.text,
                token_overflow=tokenizer.count(text) > config.chunking.max_tokens,
            )
        )
    return merge_body_peers(output, nodes, config, tokenizer)


def merge_body_peers(pieces, nodes, config, tokenizer):
    from .facts import ASSETS, BODY_KINDS

    by_id = {n["id"]: n for n in nodes}
    boundaries = {n["reading_order"] for n in nodes if n["kind"] in ASSETS}
    merged = []

    def context(piece):
        items = [by_id[nid] for nid in piece["node_ids"]]
        if not items or any(n["kind"] not in BODY_KINDS or n["reading_order"] is None for n in items):
            return None
        keys = {(n["parent_id"], n["section_id"]) for n in items}
        return (next(iter(keys)), tuple(piece["headings"])) if len(keys) == 1 else None

    for piece in pieces:
        prev = merged[-1] if merged else None
        key = context(piece)
        if prev and key is not None and context(prev) == key:
            left = max(by_id[n]["reading_order"] for n in prev["node_ids"])
            right = min(by_id[n]["reading_order"] for n in piece["node_ids"])
            body = prev["body_text"] + "\n" + piece["body_text"]
            text = "\n".join([*piece["headings"], body])
            # Same-node split fragments may be joined only if the full text fits.
            if (
                right >= left
                and not any(left <= p <= right for p in boundaries if p is not None)
                and tokenizer.count(text) <= config.chunking.max_tokens
            ):
                ids = list(dict.fromkeys([*prev["node_ids"], *piece["node_ids"]]))
                merged[-1] = make_chunk(
                    text,
                    [by_id[n] for n in ids],
                    "text",
                    piece["headings"],
                    body_text=body,
                    asset_ids=[],
                    token_overflow=False,
                )
                continue
        merged.append(piece)
    # HybridChunker may keep an indivisible wide table row with repeated headers
    # above the target. Keep it intact and expose the overflow; never silently drop cells.
    for p in merged:
        p.pop("body_text", None)
    if config.chunking.source_scope == "fragment":
        from .provenance import localize_fragment

        return [localize_fragment(p, [by_id[n] for n in p["node_ids"]]) for p in merged]
    return merged


def chunk(config: Config, store: Store, kind: str) -> dict:
    if kind == "enriched":
        from .enrichment import enriched_chunks

        base = current_index(config, store, "structured")
        pieces, enhancement_ids = enriched_chunks(config, store, base)
        parsed = base["parse_ids"]
    else:
        parses = active_parses(config, store, kind)
        tokenizer = Tokenizer(config.tokenizer)
        pieces = []
        for parsed_doc in parses:
            nodes = store.nodes(parsed_doc["id"])
            pieces.extend(
                flat_chunks(nodes, config, tokenizer)
                if kind == "flat"
                else hybrid_chunks(parsed_doc, nodes, config, tokenizer)
            )
        parsed = [p["id"] for p in parses]
        enhancement_ids = []
    if not pieces:
        raise RagError("No source-bearing chunks were produced; inspect the PDFs and parses.")
    manifest = dataset(store)
    index = {
        "kind": kind,
        "dataset_id": manifest["id"],
        "parse_ids": parsed,
        "signature": chunk_signature(config),
        "enhancement_ids": enhancement_ids,
        "synthetic": manifest["synthetic"],
        "chunk_count": len(pieces),
    }
    index["id"] = digest({**index, "chunks": pieces})
    for i, piece in enumerate(pieces):
        piece["index_id"] = index["id"]
        piece["id"] = digest([index["id"], i, piece])
    # An identical rebuild must not delete its already-paid vector mappings.
    existing = store.db.execute("SELECT id FROM indices WHERE id=?", (index["id"],)).fetchone()
    if (
        not existing
        or store.get("indices", index["id"]) != index
        or store.chunks(index["id"]) != sorted(pieces, key=lambda p: p["id"])
    ):
        store.put_index(index, pieces)
    else:
        store.set_meta("index:" + kind, index["id"])
    return {
        **index,
        "types": dict(Counter(p["kind"] for p in pieces)),
        "overflow_chunks": sum(p.get("token_overflow", False) for p in pieces),
    }


def current_index(config: Config, store: Store, kind: str) -> dict:
    iid = store.meta("index:" + kind)
    if not iid:
        raise RagError(f"Run chunk --index {kind} first.")
    index = store.get("indices", iid)
    if index["signature"] != chunk_signature(config) or index["dataset_id"] != dataset(store)["id"]:
        raise RagError(f"Index {kind} is stale after configuration/code/data changes; rebuild it.")
    parser = "flat" if kind == "flat" else "structured"
    if index["parse_ids"] != [p["id"] for p in active_parses(config, store, parser)]:
        raise RagError(f"Index {kind} is stale after parsing; rebuild it.")
    if kind == "enriched":
        from .enrichment import enrichment_key

        actual = [
            enrichment_key(config, n)
            for pid in index["parse_ids"]
            for n in store.nodes(pid)
            if n["kind"] in ("table", "picture")
        ]
        if sorted(index["enhancement_ids"]) != sorted(actual):
            raise RagError("Enrichment model/options changed; rerun enrich and chunk --index enriched.")
    return index
