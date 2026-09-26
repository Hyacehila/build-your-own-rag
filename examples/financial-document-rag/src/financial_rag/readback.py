"""Bounded SQL navigation over a particular immutable fact version."""

from .common import RagError
from .facts import BODY_KINDS


def select_nodes(store, parse_ids, anchors, mode, radius, depth):
    if mode not in {"node", "window", "parent", "children", "subtree"}:
        raise RagError("Unknown read mode.")
    if not isinstance(radius, int) or isinstance(radius, bool) or not 0 <= radius <= 8:
        raise RagError("Window radius must be an integer from 0 to 8.")
    if not isinstance(depth, int) or isinstance(depth, bool) or not 0 <= depth <= 4:
        raise RagError("Subtree depth must be an integer from 0 to 4.")
    ids = []
    for anchor in anchors:
        node = store.db.execute("SELECT * FROM doc_items WHERE id=?", (anchor,)).fetchone()
        if node is None or node["parse_id"] not in parse_ids:
            raise RagError("Node belongs to a different fact version, or facts need migration.")
        if mode == "node":
            selected = [anchor]
        elif mode == "parent":
            selected = [node["parent_id"]] if node["parent_id"] else [anchor]
        elif mode == "children":
            selected = [
                r[0]
                for r in store.db.execute(
                    "SELECT id FROM doc_items WHERE parse_id=? AND parent_id=? ORDER BY sibling_order,id",
                    (node["parse_id"], anchor),
                )
            ]
        elif mode == "subtree":
            selected = [
                r[0]
                for r in store.db.execute(
                    """
                WITH RECURSIVE tree(id,level,path) AS (
                    SELECT ?,0,'' UNION ALL
                    SELECT d.id,t.level+1,t.path||'/'||printf('%08d',d.sibling_order)
                    FROM doc_items d JOIN tree t ON d.parent_id=t.id
                    WHERE d.parse_id=? AND t.level<?
                ) SELECT id FROM tree ORDER BY path,id
                """,
                    (anchor, node["parse_id"], depth),
                )
            ]
        else:
            params = (
                node["parse_id"],
                node["section_id"],
                *sorted(BODY_KINDS),
                node["reading_order"],
                radius,
            )
            kinds = ",".join("?" for _ in BODY_KINDS)
            base = f"""SELECT id FROM doc_items WHERE parse_id=? AND section_id IS ?
                        AND kind IN ({kinds}) AND reading_order """
            before = [
                r[0] for r in store.db.execute(base + "< ? ORDER BY reading_order DESC LIMIT ?", params)
            ]
            after = [r[0] for r in store.db.execute(base + "> ? ORDER BY reading_order LIMIT ?", params)]
            selected = [*reversed(before), anchor, *after]
        ids.extend(selected)
    # Associated captions/footnotes are themselves original nodes, never synthesized facts.
    output = []
    seen = set()
    for nid in ids:
        related = [
            r[0]
            for r in store.db.execute(
                "SELECT related_id FROM item_relations WHERE item_id=? ORDER BY related_id", (nid,)
            )
        ]
        for candidate in [nid, *related]:
            if candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
    return output
