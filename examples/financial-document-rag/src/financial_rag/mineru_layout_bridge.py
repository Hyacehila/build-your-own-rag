"""Page-faithful MinerU 3.4.4 layout import before cross-page paragraph/table merging."""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    Size,
)
from PIL import Image

from .common import RagError, digest, read_json, sha256, write_json
from .mineru_bridge import html_table_data


def block_text(block):
    return "\n".join(
        " ".join(str(s.get("content", "")) for s in line.get("spans", []) if s.get("content"))
        for line in block.get("lines", [])
    ).strip()


def infer_header_rows(data):
    """Conservative, recorded header-role inference; never modify text or numbers."""
    if any(c.column_header for c in data.table_cells):
        return {"source": "explicit_html", "rows": []}
    numeric = re.compile(r"^[($\s+−-]*\d[\d,]*(?:\.\d+)?[%)\s]*$")
    years = re.compile(r"^(?:19|20)\d{2}$")
    first_body = None
    for r in range(min(data.num_rows, 8)):
        cells = [c for c in data.table_cells if c.start_row_offset_idx == r]
        values = [c.text.strip() for c in cells]
        quantities = [v for v in values if numeric.fullmatch(v) and not years.fullmatch(v)]
        first = next((c.text.strip() for c in cells if c.start_col_offset_idx == 0), "")
        if quantities and first and not re.match(r"^(?:In\s+|For\s+the\s+|Years?\b)", first, re.I):
            first_body = r
            break
    if first_body is None or first_body == 0:
        return {"source": "unresolved", "rows": []}
    for cell in data.table_cells:
        if cell.end_row_offset_idx <= first_body:
            cell.column_header = True
    return {"source": "pre_numeric_body_prefix", "rows": list(range(first_body))}


def convert_document(bundle, original, output, boundaries):
    from .outline import rebuild_scoped
    from .parsing import docling_nodes

    manifest = read_json(bundle / "manifest.json")
    results = read_json(bundle / "results.json")
    if not results["complete"] or any(r["state"] != "done" for r in results["results"]):
        raise RagError("Full comparison requires every packet to finish successfully.")
    if sha256(Path(original["pdf_path"])) != manifest["original_sha256"]:
        raise RagError("Original PDF differs from the submitted version.")
    document = DoclingDocument(name=original["doc_id"])
    with pymupdf.open(original["pdf_path"]) as pdf:
        for page in pdf:
            document.add_page(
                page_no=page.number + 1, size=Size(width=page.rect.width, height=page.rect.height)
            )
    origins = {}
    warnings = []
    headers = []
    metadata = []
    seen = set()
    continuation = []
    result_by_id = {r["data_id"]: r for r in results["results"]}
    output.mkdir(parents=True, exist_ok=True)

    for entry in manifest["files"]:
        result = result_by_id[entry["data_id"]]
        archive_path = Path(result["zip_file"])
        if sha256(archive_path) != result["sha256"]:
            raise RagError("Cloud result checksum differs.")
        with zipfile.ZipFile(archive_path) as archive:
            layout = json.loads(archive.read("layout.json"))
            if layout.get("_version_name") != "3.4.4" or len(layout.get("pdf_info", [])) != len(
                entry["pages"]
            ):
                raise RagError(
                    "Unexpected MinerU layout version or page count; update the adapter explicitly."
                )
            metadata.append({k: v for k, v in layout.items() if k != "pdf_info"})
            for provider_page, page_map in zip(layout["pdf_info"], entry["pages"]):
                if provider_page["page_idx"] != page_map["uploaded_page_index"]:
                    raise RagError("Cloud page index differs from upload map.")
                if len(provider_page["page_size"]) != 2 or any(
                    abs(a - b) > 0.1 for a, b in zip(provider_page["page_size"], page_map["page_size"])
                ):
                    raise RagError("Cloud page dimensions differ from original PDF.")
                if not page_map["owner"]:
                    continue
                number = page_map["page_number"]
                width, height = page_map["page_size"]
                if number in seen or number in manifest["excluded_page_numbers"]:
                    raise RagError("Repeated or excluded owned source page.")
                seen.add(number)

                def provenance(block, text=""):
                    b = block["bbox"]
                    if len(b) != 4 or not 0 <= b[0] < b[2] <= width or not 0 <= b[1] < b[3] <= height:
                        raise RagError(f"Invalid layout geometry on source page {number}.")
                    return ProvenanceItem(
                        page_no=number,
                        charspan=(0, len(text)),
                        bbox=BoundingBox(l=b[0], t=b[1], r=b[2], b=b[3], coord_origin=CoordOrigin.TOPLEFT),
                    )

                def asset_image(block):
                    paths = [
                        s["image_path"]
                        for line in block.get("lines", [])
                        for s in line.get("spans", [])
                        if s.get("image_path")
                    ]
                    if not paths:
                        return None
                    name = "images/" + Path(paths[0]).name
                    if name not in archive.namelist():
                        warnings.append({"page": number, "issue": "missing_asset_image", "path": name})
                        return None
                    content = archive.read(name)
                    path = output / "assets" / Path(name).name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(content)
                    with Image.open(path) as im:
                        size = Size(width=im.width, height=im.height)
                    return ImageRef(mimetype="image/jpeg", dpi=144, size=size, uri=Path("assets") / path.name)

                blocks = [(b, False) for b in provider_page["preproc_blocks"]] + [
                    (b, True) for b in provider_page.get("discarded_blocks", [])
                ]
                for ordinal, (block, discarded) in enumerate(blocks):
                    kind = block["type"]
                    text = block_text(block)
                    origin = {
                        "packet": entry["data_id"],
                        "uploaded_page_index": provider_page["page_idx"],
                        "source_page_number": number,
                        "block_index": block.get("index"),
                        "type": kind,
                        "discarded": discarded,
                    }
                    if kind in {"table", "image", "chart"}:
                        sub = block.get("blocks", [])
                        bodies = [b for b in sub if b["type"] in {"table_body", "image_body", "chart_body"}]
                        if not bodies:
                            raise RagError(f"Asset has no body on page {number}.")
                        for body in bodies:
                            if kind == "table":
                                htmls = [
                                    s["html"]
                                    for line in body.get("lines", [])
                                    for s in line.get("spans", [])
                                    if s.get("html")
                                ]
                                if len(htmls) != 1:
                                    raise RagError(f"Missing page-local table HTML on page {number}.")
                                data = html_table_data(htmls[0])
                                header = infer_header_rows(data)
                                node = document.add_table(data=data, prov=provenance(body))
                                headers.append({"ref": node.self_ref, "page": number, **header})
                                if body.get("cell_merge") or block.get("cell_merge"):
                                    continuation.append(
                                        {
                                            "ref": node.self_ref,
                                            "page": number,
                                            "provider_cell_merge": body.get(
                                                "cell_merge", block.get("cell_merge")
                                            ),
                                        }
                                    )
                            else:
                                node = document.add_picture(prov=provenance(body))
                            node.image = asset_image(body)
                            origins[node.self_ref] = origin
                            for note in sub:
                                if note["type"].endswith("_caption") or note["type"].endswith("_footnote"):
                                    note_text = block_text(note)
                                    if not note_text:
                                        continue
                                    is_caption = note["type"].endswith("_caption")
                                    related = document.add_text(
                                        label=DocItemLabel.CAPTION if is_caption else DocItemLabel.FOOTNOTE,
                                        text=note_text,
                                        prov=provenance(note, note_text),
                                    )
                                    (node.captions if is_caption else node.footnotes).append(
                                        related.get_ref()
                                    )
                                    origins[related.self_ref] = {**origin, "type": note["type"]}
                    elif kind == "title" and not discarded:
                        node = document.add_heading(
                            text=text,
                            level=max(1, min(12, int(block.get("level", 1)))),
                            prov=provenance(block, text),
                        )
                        origins[node.self_ref] = origin
                    else:
                        known = {
                            "text",
                            "ref_text",
                            "header",
                            "footer",
                            "page_number",
                            "page_footnote",
                            "footnote",
                            "interline_equation",
                            "aside_text",
                            "title",
                        }
                        if kind not in known:
                            raise RagError(f"Unsupported layout block {kind} on source page {number}.")
                        label = (
                            DocItemLabel.FORMULA
                            if kind == "interline_equation"
                            else DocItemLabel.FOOTNOTE
                            if kind in {"page_footnote", "footnote"}
                            else DocItemLabel.PAGE_HEADER
                            if kind == "header"
                            else DocItemLabel.PAGE_FOOTER
                            if kind in {"footer", "page_number"}
                            else DocItemLabel.TEXT
                        )
                        furniture = discarded and kind not in {"page_footnote", "footnote", "ref_text"}
                        if furniture and len(text) > 200 and block["bbox"][3] > height * 0.2:
                            warnings.append(
                                {
                                    "page": number,
                                    "issue": "substantial_text_classified_as_furniture",
                                    "characters": len(text),
                                    "type": kind,
                                }
                            )
                        node = document.add_text(
                            label=label,
                            text=text,
                            prov=provenance(block, text),
                            content_layer=ContentLayer.FURNITURE if furniture else ContentLayer.BODY,
                        )
                        origins[node.self_ref] = origin
    if len(seen) != manifest["unique_body_pages"]:
        raise RagError("Converted source page count differs from submitted body.")
    rebuild_scoped(document, boundaries)
    pid = digest(
        ["mineru-page-local-layout-v1", manifest["identity"], [r["sha256"] for r in results["results"]]]
    )
    write_json(output / "document.json", document.model_dump(mode="json"))
    nodes = docling_nodes(document, original, pid)
    write_json(output / "nodes.json", nodes)
    report = {
        "contract": "mineru-page-local-layout-v1",
        "pages": len(seen),
        "tables": len(document.tables),
        "pictures": len(document.pictures),
        "nodes": len(nodes),
        "metadata": metadata,
        "warnings": warnings,
        "header_roles": headers,
        "continuation_candidates": continuation,
        "origins": origins,
        "source_page_numbers_preserved": True,
        "cell_bboxes_available": False,
        "scope_boundaries": boundaries,
        "active_database_changed": False,
        "header_role_counts": dict(Counter(h["source"] for h in headers)),
    }
    write_json(output / "report.json", report)
    return {k: v for k, v in report.items() if k not in {"origins", "header_roles"}}
