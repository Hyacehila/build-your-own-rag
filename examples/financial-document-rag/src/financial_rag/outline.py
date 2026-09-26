"""Reviewed style profiles and auditable outline edits; no source-text rewriting."""

from bisect import bisect_right
from collections import Counter

from docling_core.types.doc import GroupItem, GroupLabel, SectionHeaderItem, TitleItem

from .common import RagError, digest


def fingerprint_content(document):
    def clean(value):
        if isinstance(value, list):
            return [clean(x) for x in value]
        if isinstance(value, dict):
            ignored = {"self_ref", "parent", "children", "level", "label", "content_layer"}
            return {k: clean(v) for k, v in value.items() if k not in ignored}
        return value

    # Structural groups may be added; actual content objects must be unchanged.
    return digest(
        [
            clean([x.model_dump(mode="json") for x in getattr(document, k)])
            for k in ("texts", "tables", "pictures", "key_value_items", "form_items")
        ]
    )


def rebuild_scoped(document, boundaries, former_heading_refs=None):
    """Reset the heading stack at source-document boundaries, retaining asset/list children."""
    flow = []
    heading_refs = {x.self_ref for x in document.texts if isinstance(x, (SectionHeaderItem, TitleItem))}
    heading_refs.update(former_heading_refs or [])
    existing_groups = {}

    def has_nested_heading(item, seen=()):
        if item.self_ref in heading_refs:
            return True
        if item.self_ref in seen:
            raise RagError("Cycle in document structure.")
        return any(has_nested_heading(ref.resolve(document), (*seen, item.self_ref)) for ref in item.children)

    def visit(item):
        # Docling sometimes keeps a real section heading inside a form/key-value group. Leaving that
        # heading in the group means it never reaches the level stack, so the following body content
        # can remain attached to the previous section. Flatten only containers that actually contain a
        # heading; ordinary list/table/picture containers remain intact.
        if isinstance(item, GroupItem) and has_nested_heading(item):
            children = list(item.children)
            item.children = []
            # The container is no longer part of the rebuilt tree. Leaving its old parent pointer
            # would create a parent/children disagreement in the relational projection.
            item.parent = None
            for ref in children:
                visit(ref.resolve(document))
            return
        flow.append(item)
        if item.self_ref in heading_refs:
            children = list(item.children)
            item.children = [r for r in children if getattr(r.resolve(document), "label", None) == "inline"]
            for ref in children:
                if ref not in item.children:
                    visit(ref.resolve(document))

    # Existing scope wrappers are implementation containers, not extra evidence.
    for ref in list(document.body.children):
        item = ref.resolve(document)
        if getattr(item, "name", "").startswith("source-range:"):
            existing_groups[int(item.name.split(":", 1)[1])] = item
            for child in list(item.children):
                visit(child.resolve(document))
            item.children = []
        else:
            visit(item)
    document.body.children = []
    starts = sorted(set(boundaries))
    groups = {}
    scopes = {}
    stacks = {}
    current = starts[0]
    for item in flow:
        prov = getattr(item, "prov", [])
        if prov:
            current = starts[max(0, bisect_right(starts, min(p.page_no for p in prov)) - 1)]
        if current not in groups:
            if current in existing_groups:
                groups[current] = existing_groups[current]
                document.body.children.append(groups[current].get_ref())
            else:
                groups[current] = document.add_group(label=GroupLabel.SECTION, name=f"source-range:{current}")
            stacks[current] = [(0, groups[current])]
        stack = stacks[current]
        level = (
            item.level if isinstance(item, SectionHeaderItem) else 1 if isinstance(item, TitleItem) else None
        )
        if level is not None:
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()
        parent = stack[-1][1]
        item.parent = parent.get_ref()
        parent.children.append(item.get_ref())
        scopes[item.self_ref] = current
        if level is not None:
            stack.append((level, item))
    return scopes


def apply_outline_patches(document, proposal, boundaries):
    """Nest existing heading subtrees only. Reject invalid/forward/cross-document parents."""
    doc = document.model_copy(deep=True)
    before = fingerprint_content(doc)
    headings = [h for h in doc.texts if isinstance(h, SectionHeaderItem)]
    order = {h.self_ref: i for i, h in enumerate(headings)}
    by_ref = {h.self_ref: h for h in headings}
    starts = sorted(boundaries)

    def scope(h):
        return starts[max(0, bisect_right(starts, h.prov[0].page_no) - 1)]

    # Freeze pre-edit subtrees; earlier edits must not silently widen later edit ranges.
    members = {}
    for i, h in enumerate(headings):
        members[h.self_ref] = [h]
        for next_h in headings[i + 1 :]:
            if scope(next_h) != scope(h) or next_h.level <= h.level:
                break
            members[h.self_ref].append(next_h)
    patches = proposal.get("nest", [])
    if len(patches) > 120 or len({p["child_ref"] for p in patches}) != len(patches):
        raise RagError("Duplicate or excessive outline edits.")
    applied = []
    for patch in sorted(patches, key=lambda p: order.get(p.get("child_ref"), -1)):
        child = by_ref.get(patch.get("child_ref"))
        parent = by_ref.get(patch.get("parent_ref"))
        if (
            not child
            or not parent
            or order[parent.self_ref] >= order[child.self_ref]
            or scope(child) != scope(parent)
        ):
            raise RagError("Outline edit needs existing, preceding headings in the same source document.")
        evidence = patch.get("evidence_pages", [])
        if not evidence or any(type(p) is not int or p not in doc.pages for p in evidence):
            raise RagError("Outline edit lacks valid source-page evidence.")
        delta = parent.level + 1 - child.level
        for h in members[child.self_ref]:
            h.level += delta
            if not 1 <= h.level <= 12:
                raise RagError("Outline edit exceeds the structural depth bound.")
        applied.append({**patch, "delta": delta, "affected_headings": len(members[child.self_ref])})
    rebuild_scoped(doc, boundaries)
    # Check that level edits actually achieved the requested parent, without a shadowing peer.
    for patch in patches:
        if by_ref[patch["child_ref"]].parent.cref != patch["parent_ref"]:
            raise RagError("Requested outline parent was shadowed by an intervening section.")
    if fingerprint_content(doc) != before:
        raise RagError("Outline edits altered source content.")
    return doc, {
        "applied": applied,
        "content_unchanged": True,
        "level_counts": dict(Counter(h.level for h in headings)),
    }


def apply_item_reparents(document, moves, boundaries):
    """Move an existing item/subtree under an earlier heading without changing source content.

    This is deliberately narrower than a general tree editor. It exists for a verified layout-reading-order
    defect where a table, figure, body leaf or already correctly leveled heading subtree follows a later
    heading in parser order but belongs to an earlier source section. A heading move may only establish
    the direct parent implied by its existing level; level changes remain the responsibility of
    ``apply_outline_patches``.
    """
    if not moves:
        return document, {"applied": []}
    if len(moves) > 120 or len({m.get("child_ref") for m in moves}) != len(moves):
        raise RagError("Duplicate or excessive item reparent edits.")
    doc = document.model_copy(deep=True)
    before = fingerprint_content(doc)
    items = {
        item.self_ref: item
        for name in ("texts", "tables", "pictures", "groups")
        for item in getattr(doc, name)
    }
    starts = sorted(boundaries)

    def scope(item):
        if not getattr(item, "prov", None):
            raise RagError("Item reparent needs source provenance.")
        return starts[max(0, bisect_right(starts, min(p.page_no for p in item.prov)) - 1)]

    def descendants(item):
        result, stack = set(), list(item.children)
        while stack:
            ref = stack.pop()
            if ref.cref in result:
                continue
            result.add(ref.cref)
            stack.extend(items[ref.cref].children)
        return result

    applied = []
    for move in moves:
        child, parent = items.get(move.get("child_ref")), items.get(move.get("parent_ref"))
        evidence = move.get("evidence_pages", [])
        if (
            child is None
            or parent is None
            or not isinstance(parent, (SectionHeaderItem, TitleItem))
            or child.self_ref == parent.self_ref
            or parent.self_ref in descendants(child)
            or scope(child) != scope(parent)
            or not evidence
            or any(type(page) is not int or page not in doc.pages for page in evidence)
            or (isinstance(child, (SectionHeaderItem, TitleItem)) and child.level != parent.level + 1)
        ):
            raise RagError(
                "Item reparent needs a same-scope child, heading parent, valid evidence and unchanged heading level."
            )
        old_parent = child.parent.resolve(doc)
        if old_parent.self_ref == parent.self_ref:
            raise RagError("Item reparent is redundant.")
        old_parent.children = [ref for ref in old_parent.children if ref.cref != child.self_ref]
        parent.children.append(child.get_ref())
        child.parent = parent.get_ref()
        applied.append(
            {
                **move,
                "old_parent_ref": old_parent.self_ref,
                "new_parent_ref": parent.self_ref,
            }
        )
    if fingerprint_content(doc) != before:
        raise RagError("Item reparent altered source content.")
    return doc, {"applied": applied, "content_unchanged": True}
