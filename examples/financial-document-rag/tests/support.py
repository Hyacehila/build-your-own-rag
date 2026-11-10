"""Small temporary inputs for testing storage, parsing and retrieval behavior."""

from types import SimpleNamespace

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from financial_rag.common import digest, write_json, write_jsonl
from financial_rag.parsing import parse


def prepare_test_source(config, store, monkeypatch):
    """Build only temporary, generic inputs; no tutorial or benchmark data is shipped."""
    root = config.work_dir / "test-input"
    root.mkdir(parents=True, exist_ok=True)
    pdf_path = root / "input.pdf"
    rows = [["Name", "Column A", "Column B"]]
    rows.extend([[f"item {i:02d}", str(i), "9001" if i == 24 else str(i + 1)] for i in range(1, 26)])
    with pymupdf.open() as pdf:
        for page_no in range(1, 4):
            page = pdf.new_page(width=612, height=792)
            page.insert_text((40, 42), f"Section {page_no}: test source")
            if page_no == 2:
                page.insert_text((40, 68), "Table A")
                for i, row in enumerate(rows):
                    y = 88 + i * 22
                    for j, value in enumerate(row):
                        x0, x1 = [40, 300, 430, 572][j:j + 2]
                        page.draw_rect((x0, y, x1, y + 22), color=(0.5, 0.5, 0.5))
                        page.insert_text((x0 + 5, y + 15), value, fontsize=10)
            else:
                page.insert_text((40, 72), "alpha item 24 and related context")
                page.draw_rect((40, 110, 300, 250))
        pdf.save(pdf_path)

    doc_id = "test-source"
    version = digest(["test-source", str(config.work_dir)])
    source = {
        "id": version,
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "sha256": version,
        "pages": 3,
        "dataset_revision": "test-input",
        "synthetic": True,
    }
    store.put("documents", version, source)

    # A few independent rows exercise runner, retry, scoring and page bindings.
    queries = [{"query_id": f"q{i}", "query": f"Find item {i}"} for i in range(1, 4)]
    labels = [
        {
            "query_id": q["query_id"],
            "answer": f"value {i}",
            "raw_answers": [],
            "content_type": ["text"],
            "query_types": ["lookup"],
            "query_format": "short_answer",
            "qrels": [{"doc_id": doc_id, "doc_version": version, "page_index": i - 1,
                       "page_number": i, "grade": 2}],
        }
        for i, q in enumerate(queries, 1)
    ]
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "labels.jsonl", labels)
    manifest = {
        "id": version,
        "repo": "test-input",
        "revision": "test-input",
        "synthetic": True,
        "language": "english",
        "documents": [source],
        "counts": {"documents": 1, "pages": 3, "queries": len(queries)},
        "prepared_dir": str(root),
    }
    write_json(root / "manifest.json", manifest)
    store.set_meta("dataset", manifest)

    document = DoclingDocument(name=doc_id)
    for page_no in range(1, 4):
        document.add_page(page_no=page_no, size=Size(width=612, height=792))

    def prov(page_no, box, text=""):
        return ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, len(text)),
        )

    parent = document.add_heading(text="Section A", level=1, prov=prov(1, [40, 25, 300, 50]))
    text = "alpha source context for a retrieval and readback test"
    document.add_text(label=DocItemLabel.TEXT, text=text, parent=parent, prov=prov(1, [40, 60, 500, 85], text))
    caption = document.add_text(label=DocItemLabel.CAPTION, text="Figure A", parent=parent,
                                prov=prov(1, [40, 260, 300, 280]))
    document.add_picture(caption=caption, parent=parent, prov=prov(1, [40, 110, 300, 250]))
    section = document.add_heading(text="Section B", level=2, parent=parent, prov=prov(2, [40, 25, 300, 50]))
    table_caption = document.add_text(label=DocItemLabel.CAPTION, text="Table A", parent=section,
                                      prov=prov(2, [40, 54, 300, 74]))
    cells = [
        TableCell(text=value, start_row_offset_idx=i, end_row_offset_idx=i + 1,
                  start_col_offset_idx=j, end_col_offset_idx=j + 1, column_header=(i == 0))
        for i, row in enumerate(rows) for j, value in enumerate(row)
    ]
    document.add_table(data=TableData(num_rows=len(rows), num_cols=3, table_cells=cells),
                       caption=table_caption, parent=section, prov=prov(2, [40, 88, 572, 660]))
    last = document.add_heading(text="Section C", level=1, prov=prov(3, [40, 25, 300, 50]))
    text = "additional test source context for the last page"
    document.add_text(label=DocItemLabel.TEXT, text=text, parent=last, prov=prov(3, [40, 60, 500, 85], text))

    class Converter:
        format_to_options = {InputFormat.PDF: SimpleNamespace(
            pipeline_options=SimpleNamespace(model_dump=lambda **_: {})
        )}

        def convert(self, path, **kwargs):
            return SimpleNamespace(document=document, status=SimpleNamespace(value="success"), errors=[])

    with monkeypatch.context() as patch:
        patch.setattr("financial_rag.parsing._converter", lambda _: Converter())
        assert not parse(config, store, "structured")["failed"]
    assert not parse(config, store, "flat")["failed"]
    return source
