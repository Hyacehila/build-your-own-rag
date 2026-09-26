"""Reviewed empty-table to diagram conversion, preserving source geometry and links."""

from docling_core.types.doc import DoclingDocument, PictureItem

from .common import RagError


def empty_tables_as_pictures(document, refs):
    if not refs:
        return document, {}, []
    wanted = set(refs)
    if len(wanted) != len(refs):
        raise RagError("Duplicate asset reclassification.")
    selected = [t for t in document.tables if t.self_ref in wanted]
    if len(selected) != len(wanted) or any(c.text.strip() for t in selected for c in t.data.table_cells):
        raise RagError("Only existing empty table facts may be reclassified as pictures.")
    snapshot = document.model_dump(mode="json")
    mapping = {}
    decisions = []
    for table in selected:
        old = table.model_dump(mode="json")
        ref = f"#/pictures/{len(snapshot['pictures'])}"
        picture = PictureItem(
            self_ref=ref,
            parent=table.parent,
            children=table.children,
            prov=table.prov,
            captions=table.captions,
            footnotes=table.footnotes,
            references=table.references,
            content_layer=table.content_layer,
            image=table.image,
        )
        snapshot["pictures"].append(picture.model_dump(mode="json"))
        mapping[table.self_ref] = ref
        decisions.append(
            {
                "before_ref": table.self_ref,
                "after_ref": ref,
                "pages": [p.page_no for p in table.prov],
                "original": old,
                "reason": "Source-reviewed diagram; original table had no cell text.",
            }
        )
    retained = [t for t in snapshot["tables"] if t["self_ref"] not in wanted]
    mapping.update({t["self_ref"]: f"#/tables/{i}" for i, t in enumerate(retained)})
    snapshot["tables"] = retained

    def remap(value):
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {
                k: mapping.get(v, v) if k in {"self_ref", "cref"} and isinstance(v, str) else remap(v)
                for k, v in value.items()
            }
        return value

    return DoclingDocument.model_validate(remap(snapshot)), mapping, decisions
