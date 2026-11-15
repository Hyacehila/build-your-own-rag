"""MinerU structured results -> DoclingDocument, preserving original PDF identity.

The cloud trial returns legacy content_list + layout JSON (3.4.4). Parse that
observed contract explicitly, rather than assuming the latest open-source 4.x
schema is identical. Keep archives for replay and reject unsupported shapes.
"""

import math

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from lxml import html

from .common import RagError

CONTRACT = "mineru-content-list-v1-to-docling-v1"


def html_table_data(source):
    if len(source) > 5_000_000:
        raise RagError("MinerU table HTML exceeds the conversion limit.")
    tree = html.fromstring(source)
    if tree.tag.lower() != "table" or tree.xpath(".//table"):
        raise RagError("Expected one flat HTML table; nested tables require an explicit adapter.")
    rows = tree.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    cells = []
    occupied = set()
    columns = 0

    def visible(node):
        value = node.text or ""
        for child in node:
            if child.tag in {"br", "p", "li", "div"}:
                value += "\n"
            value += visible(child) + (child.tail or "")
        return value

    for r, row in enumerate(rows):
        col = 0
        for cell in row.xpath("./td|./th"):
            while (r, col) in occupied:
                col += 1
            try:
                rs = int(cell.get("rowspan", "1"))
                cs = int(cell.get("colspan", "1"))
            except ValueError as e:
                raise RagError("Invalid table span in MinerU HTML.") from e
            if not 1 <= rs <= 1000 or not 1 <= cs <= 1000 or r + rs > len(rows) or col + cs > 1000:
                raise RagError("MinerU table span is outside the table geometry.")
            slots = {(rr, cc) for rr in range(r, r + rs) for cc in range(col, col + cs)}
            if occupied & slots:
                raise RagError("Overlapping MinerU table cells.")
            occupied.update(slots)
            cells.append(
                TableCell(
                    text=visible(cell).strip(),
                    row_span=rs,
                    col_span=cs,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + rs,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + cs,
                    column_header=cell.tag.lower() == "th",
                )
            )
            col += cs
            columns = max(columns, col)
    if not rows or not columns or len(cells) > 100_000:
        raise RagError("Empty or oversized MinerU table.")
    return TableData(num_rows=len(rows), num_cols=columns, table_cells=cells)


def convert_content_list(items, mapping):
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise RagError("Expected flat MinerU content_list v1 with type/page_idx/bbox.")
    width, height = mapping["page_size"]
    page_no = mapping["source_page_index"] + 1
    doc = DoclingDocument(name=mapping["doc_id"])
    doc.add_page(page_no=page_no, size=Size(width=width, height=height))
    warnings = []
    origins = {}
    page_only = []

    def prov(item, text="", page_level=False):
        if item.get("page_idx") != mapping["uploaded_page_index"]:
            raise RagError("MinerU page index does not match the explicit uploaded-page map.")
        box = item.get("bbox")
        if page_level:
            box = [0, 0, 1000, 1000]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(n, (int, float)) or not math.isfinite(n) or not 0 <= n <= 1000 for n in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise RagError("Missing/invalid normalized block bbox; Markdown cannot supply PDF provenance.")
        return ProvenanceItem(
            page_no=page_no,
            charspan=(0, len(text)),
            bbox=BoundingBox(
                l=box[0] * width / 1000,
                t=box[1] * height / 1000,
                r=box[2] * width / 1000,
                b=box[3] * height / 1000,
                coord_origin=CoordOrigin.TOPLEFT,
            ),
        )

    for index, item in enumerate(items):
        kind = item.get("type")
        text = item.get("text", "")
        if kind in {"text", "header", "footer", "page_number", "aside_text"}:
            furniture = kind != "text"
            label = (
                DocItemLabel.PAGE_HEADER
                if kind == "header"
                else DocItemLabel.PAGE_FOOTER
                if kind in {"footer", "page_number"}
                else DocItemLabel.TEXT
            )
            if not furniture and item.get("text_level", 0) > 0:
                node = doc.add_heading(
                    text=text, level=min(6, int(item["text_level"])), prov=prov(item, text)
                )
            else:
                node = doc.add_text(
                    label=label,
                    text=text,
                    prov=prov(item, text),
                    content_layer=ContentLayer.FURNITURE if furniture else ContentLayer.BODY,
                )
        elif kind in {"table", "image"}:
            captions = item.get("table_caption" if kind == "table" else "image_caption", [])
            caption = None
            if captions:
                caption_text = "\n".join(captions)
                caption = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=caption_text,
                    prov=prov(item, caption_text, page_level=True),
                )
                page_only.append(caption.self_ref)
                origins[caption.self_ref] = f"content_list/{index}/caption"
            if kind == "table":
                node = doc.add_table(
                    data=html_table_data(item.get("table_body", "")), caption=caption, prov=prov(item)
                )
                if not any(c.column_header for c in node.data.table_cells):
                    warnings.append(
                        {"ref": node.self_ref, "issue": "html_has_no_explicit_column_header_flags"}
                    )
            else:
                node = doc.add_picture(caption=caption, prov=prov(item))
                if not captions:
                    warnings.append({"ref": node.self_ref, "issue": "picture_has_no_searchable_caption"})
            for footnote in item.get("table_footnote" if kind == "table" else "image_footnote", []):
                note = doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=footnote, prov=prov(item, footnote, page_level=True)
                )
                node.footnotes.append(note.get_ref())
                page_only.append(note.self_ref)
                origins[note.self_ref] = f"content_list/{index}/footnote"
        elif kind in {"equation", "interline_equation"}:
            node = doc.add_text(label=DocItemLabel.FORMULA, text=text, prov=prov(item, text))
        else:
            raise RagError(f"Unsupported MinerU content type {str(kind)[:40]}; raw result retained.")
        origins[node.self_ref] = f"content_list/{index}"
    # Use the producer's explicit heading levels. Never re-run Docling's PDF
    # numbering/font inference over MinerU output or use Markdown as input.
    doc._hierarchize()
    return doc, {
        "contract": CONTRACT,
        "source_mapping": mapping,
        "json_origins": origins,
        "page_only_refs": page_only,
        "warnings": warnings,
        "cell_bbox_available": False,
        "bbox_units": "content_list: 0..1000 -> original PDF points, top-left",
    }
