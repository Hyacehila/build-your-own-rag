"""Relational projection of document facts and their explicit structure."""

from __future__ import annotations

import copy

from financial_rag.common import RagError

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
