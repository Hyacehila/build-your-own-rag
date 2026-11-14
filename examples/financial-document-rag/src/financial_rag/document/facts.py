"""Immutable raw parses, separately versioned hierarchy, and relational fact projection."""

from __future__ import annotations

import copy
from pathlib import Path

from financial_rag.common import RagError, digest, sha256, versions, write_json

NORMALIZATION = {
    "contract": "existing-heading-levels-v1",
    "projection": "content-layers-v1",
    **versions("docling-core"),
}
HEADINGS = {"title", "section_header"}
BODY_KINDS = {"text", "paragraph", "list_item", "formula", "code"}
ASSETS = {"table", "picture"}


def relations(nodes: list[dict]) -> list[dict]:
    """Project explicit ancestry without inventing a title or altering raw node data."""
    result = copy.deepcopy(nodes)
    refs = {n["ref"]: n for n in result}
    if len(refs) != len(result):
        raise RagError("Duplicate fact references.")
    chains = {}

    def ancestors(ref, visiting=()):
        if ref in chains:
            return chains[ref]
        if ref in visiting or ref not in refs:
            raise RagError("Cyclic or unresolved fact parent.")
        parent = refs[ref].get("parent_ref")
        chain = [*ancestors(parent, (*visiting, ref)), parent] if parent else []
        chains[ref] = chain
        return chain

    order = 0
    for node in result:
        chain = ancestors(node["ref"])
        parent = refs.get(node.get("parent_ref"))
        node["parent_id"] = parent["id"] if parent else None
        node["depth"] = len(chain)
        siblings = parent.get("children", []) if parent else [node["ref"]]
        if node["ref"] not in siblings:
            raise RagError("Parent/child references disagree.")
        node["sibling_order"] = siblings.index(node["ref"])
        heading_refs = [
            r
            for r in [*chain, node["ref"]]
            if refs[r]["kind"] in HEADINGS and refs[r].get("raw", {}).get("content_layer", "body") == "body"
        ]
        node["section_id"] = refs[heading_refs[-1]]["id"] if heading_refs else None
        node["section_ref"] = heading_refs[-1] if heading_refs else None
        node["headings"] = [refs[r]["text"] for r in heading_refs]
        # Containers and content nested inside an asset remain addressable, but are
        # not independent body-reading positions. Furniture never joins a body window.
        readable = (
            node.get("raw", {}).get("content_layer", "body") == "body"
            and bool(node.get("sources"))
            and not any(refs[r]["kind"] in ASSETS for r in chain)
        )
        node["reading_order"] = order if readable else None
        if readable:
            order += 1
        for child in node.get("children", []):
            if child not in refs or refs[child].get("parent_ref") != node["ref"]:
                raise RagError("Unresolved or inconsistent fact child.")
    return result


def fact_content(document: dict) -> dict:
    """Everything except the two fields this normalization is allowed to modify."""

    def strip(value):
        if isinstance(value, list):
            return [strip(v) for v in value]
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in {"parent", "children"}}
        return value

    return strip(document)


def normalize_document(document):
    before = fact_content(document.model_dump(mode="json"))
    normalized = document.model_copy(deep=True)
    normalized._hierarchize()  # Pinned docling-core API, guarded by complete content invariance.
    after = fact_content(normalized.model_dump(mode="json"))
    if before != after:
        raise RagError("Hierarchy normalization altered content or provenance; no facts saved.")
    return normalized, digest(before)


def normalized_parse(config, store, raw: dict, doc: dict) -> dict:
    from financial_rag.document.parsing import docling_nodes, load_docling

    if raw["status"] != "success":
        raise RagError("Cannot normalize an incomplete raw parse.")
    pid = digest(
        {"raw_parse_id": raw["id"], "raw_sha256": raw["snapshot_sha256"], "normalization": NORMALIZATION}
    )
    found = store.db.execute("SELECT id FROM parses WHERE id=?", (pid,)).fetchone()
    if found:
        parsed = store.get("parses", pid)
        if (
            Path(parsed["snapshot"]).exists()
        ):
            store.set_meta(f"parse:structured:{doc['id']}", pid)
            return parsed
    document, invariant = normalize_document(load_docling(raw))
    path = store.root / "parses" / pid / "document.json"
    write_json(path, document.model_dump(mode="json"))
    parsed = {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"resolved_options", "elapsed_seconds"}}
    parsed.update(
        id=pid,
        raw_parse_id=raw["id"],
        normalization=NORMALIZATION,
        content_invariant_sha256=invariant,
        snapshot=str(path),
        snapshot_sha256=sha256(path),
    )
    store.save_snapshot(Path(raw["snapshot"]), raw["snapshot_sha256"])
    store.save_snapshot(path, parsed["snapshot_sha256"])
    store.put_parse(parsed, docling_nodes(document, doc, pid))
    store.set_meta(f"parse:structured:{doc['id']}", pid)
    return parsed
