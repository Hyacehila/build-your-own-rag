"""Reviewed page exclusions, preserving original physical page identities.

This works on saved parser facts before rebuilding a heading tree. It neither
deletes pages from the source PDF nor silently drops a node spanning a boundary.
"""

from copy import deepcopy

from docling_core.types.doc import DoclingDocument

from .common import RagError, digest

CONTRACT = "reviewed-page-exclusions-v1"
ITEM_LISTS = ("groups", "texts", "pictures", "tables", "key_value_items", "form_items")


def item_content(item):
    """Content/provenance invariant independent of renumbered node references."""
    return {
        k: v
        for k, v in item.items()
        if k not in {"self_ref", "parent", "children", "captions", "footnotes", "references", "level"}
    }


def exclude_pages(document, page_numbers):
    excluded = set(page_numbers)
    if not excluded or any(type(p) is not int or p not in document.pages for p in excluded):
        raise RagError("Exclusions must name existing 1-based physical PDF pages.")
    snapshot = document.model_dump(mode="json")
    items = {x["self_ref"]: x for name in ITEM_LISTS for x in snapshot.get(name, [])}
    removed = set()
    for ref, item in items.items():
        pages = {p["page_no"] for p in item.get("prov", [])}
        if pages & excluded and pages - excluded:
            raise RagError(f"Node {ref} spans excluded and retained pages; split it before exclusion.")
        if pages and pages <= excluded:
            removed.add(ref)
    # Empty containers must not leave phantom sections/list boundaries.
    while True:
        empty = {
            x["self_ref"]
            for x in snapshot.get("groups", [])
            if x.get("children") and all(c["cref"] in removed for c in x["children"])
        }
        if empty <= removed:
            break
        removed |= empty
    result = deepcopy(snapshot)
    mapping = {"#/body": "#/body", "#/furniture": "#/furniture"}
    for name in ITEM_LISTS:
        result[name] = [x for x in result.get(name, []) if x["self_ref"] not in removed]
        mapping.update({x["self_ref"]: f"#/{name}/{i}" for i, x in enumerate(result[name])})

    def retained_children(refs):
        children = []
        for ref in refs:
            key = ref["cref"]
            if key in removed:
                children.extend(retained_children(items[key].get("children", [])))
            else:
                children.append(ref)
        return children

    nodes = [result["body"], result["furniture"], *[x for n in ITEM_LISTS for x in result[n]]]
    for item in nodes:
        item["children"] = retained_children(item.get("children", []))
        parent = item.get("parent")
        visited = set()
        while parent and parent["cref"] in removed:
            if parent["cref"] in visited:
                raise RagError("Cycle in source parents.")
            visited.add(parent["cref"])
            parent = items[parent["cref"]].get("parent")
        if "parent" in item:
            item["parent"] = parent

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(v) for v in value if not (isinstance(v, dict) and v.get("cref") in removed)]
        if isinstance(value, dict):
            return {k: mapping[v] if k in {"self_ref", "cref"} else rewrite(v) for k, v in value.items()}
        return value

    result = rewrite(result)
    candidate = DoclingDocument.model_validate(result)
    actual = {x["self_ref"]: x for name in ITEM_LISTS for x in result[name]}
    for old, new in mapping.items():
        if old in items and digest(item_content(items[old])) != digest(item_content(actual[new])):
            raise RagError("Page exclusion changed retained text, table data or provenance.")
    if set(candidate.pages) != set(document.pages):
        raise RagError("Page exclusion changed physical page identities.")
    return candidate, {
        "contract": CONTRACT,
        "excluded_page_numbers": sorted(excluded),
        "removed_refs": sorted(removed),
        "old_to_new_refs": mapping,
        "retained_content_unchanged": True,
        "original_page_numbers_preserved": True,
    }
