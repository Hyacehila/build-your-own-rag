from types import SimpleNamespace as NS

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    TextItem,
)

from financial_rag.chunking import chunk_signature, hybrid_chunks
from financial_rag.common import digest, write_json
from financial_rag.config import Config
from financial_rag.diagnostics.parsing_quality import inspect_snapshot
from financial_rag.parsing import active_parses, docling_nodes, load_docling, parse_sample, parse_signature
from financial_rag.parsing_guards import GuardedReadingOrderModel, merge_rejection, order_page_zones
from financial_rag.tokenization import Tokenizer


def element(cid, left, top, right, bottom, label=DocItemLabel.TEXT):
    return NS(
        cid=cid,
        l=left,
        t=top,
        r=right,
        b=bottom,
        label=label,
        page_no=1,
        page_size=Size(width=600, height=800),
    )


def test_full_width_zones_preserve_nodes_and_within_zone_order():
    left_top = element(0, 30, 730, 280, 500)
    left_bottom = element(1, 30, 300, 280, 60)
    right_top = element(2, 320, 730, 570, 500)
    right_bottom = element(3, 320, 300, 570, 60)
    table = element(4, 30, 450, 570, 350, DocItemLabel.TABLE)
    original = [left_top, left_bottom, right_top, right_bottom, table]
    ordered, changes = order_page_zones(original)
    assert [i.cid for i in ordered] == [0, 2, 4, 1, 3]
    assert len(changes) == 1 and len({id(i) for i in ordered}) == len(original)
    assert merge_rejection(left_bottom, right_top, original) == "crosses_full_width_asset"
    assert merge_rejection(left_top, right_top, original) is None
    straddler = element(5, 30, 480, 280, 300)
    assert order_page_zones([*original, straddler]) == ([*original, straddler], [])
    assert order_page_zones([left_top, right_top]) == ([left_top, right_top], [])


def test_literal_hyphen_and_original_offsets_on_different_page_sizes():
    guard = object.__new__(GuardedReadingOrderModel)
    guard.page_heights = {2: 900}
    bbox = BoundingBox(l=20, t=30, r=100, b=50, coord_origin=CoordOrigin.TOPLEFT)
    first = NS(label=DocItemLabel.TEXT)
    second = NS(
        label=DocItemLabel.TEXT, page_no=2, text="currency item", hyperlink=None, cluster=NS(bbox=bbox)
    )
    raw = "• functional-"
    item = TextItem(
        self_ref="#/texts/0",
        label=DocItemLabel.TEXT,
        text="functional-",
        orig=raw,
        prov=[ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(raw)))],
    )
    guard._merge_elements(first, second, item, 800)
    assert item.orig == "• functional-currency item"
    assert item.text == "functional-currency item"
    assert item.orig[slice(*item.prov[1].charspan)] == "currency item"
    assert item.prov[1].bbox.t == 870  # page 2 height, not the preceding page's height
    assert item.prov[1].charspan[0] == item.prov[0].charspan[1]


def test_multirow_headers_repeat_without_altering_facts_or_losing_rows(tmp_path):
    config = Config(work_dir=tmp_path)
    config.chunking.max_tokens = 150
    config.chunking.table_headers = "multirow"
    doc = DoclingDocument(name="multirow-header-fixture")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    heading = doc.add_heading(
        text="Financial statements and supplementary detailed operating disclosures", level=1
    )
    heading = doc.add_heading(
        text="Annual financial comparison for multiple reportable business divisions", level=2, parent=heading
    )
    rows = [["Measure", "Income", "Income"], ["USD million", "2024", "2023"]]
    rows += [[f"Metric_{i:02d}", f"{i}.01", f"{i}.02"] for i in range(30)]
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r < 2,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    doc.add_table(
        data=TableData(num_rows=len(rows), num_cols=3, table_cells=cells),
        parent=heading,
        prov=ProvenanceItem(
            page_no=1,
            charspan=(0, 0),
            bbox=BoundingBox(l=20, t=40, r=580, b=750, coord_origin=CoordOrigin.TOPLEFT),
        ),
    )
    before = doc.model_dump(mode="json")
    snapshot = tmp_path / "document.json"
    write_json(snapshot, before)
    source = {"id": "fixture-version", "doc_id": "fixture", "pages": 1}
    nodes = docling_nodes(doc, source, "fixture-parse")
    tokenizer = Tokenizer(config.tokenizer)
    chunks = hybrid_chunks({"snapshot": str(snapshot)}, nodes, config, tokenizer)
    assert len(chunks) > 3
    for chunk in chunks:
        assert all(term in chunk["text"] for term in ("2024", "2023", "USD million"))
        assert tokenizer.count(chunk["text"]) <= 150
    combined = "\n".join(c["text"] for c in chunks)
    assert all(combined.count(f"Metric_{i:02d}") == 1 for i in range(30))
    assert doc.model_dump(mode="json") == before
    # Bad parser metadata must not turn an entire table into an empty body.
    for cell in doc.tables[0].data.table_cells:
        cell.column_header = True
    write_json(snapshot, doc.model_dump(mode="json"))
    nodes = docling_nodes(doc, source, "fixture-parse")
    fallback = hybrid_chunks({"snapshot": str(snapshot)}, nodes, config, tokenizer)
    assert fallback
    combined = "\n".join(c["text"] for c in fallback)
    assert all(combined.count(f"Metric_{i:02d}") == 1 for i in range(30))


def test_new_options_preserve_legacy_cache_identity_and_invalidate_opt_in():
    config = Config()
    legacy_parse = parse_signature(config, "structured")
    legacy_chunk = chunk_signature(config)
    assert legacy_chunk["chunking"] == {"max_tokens": 800, "overlap": 100}
    assert not {"guarded_merges", "backend", "ocr_engine"} & legacy_parse["options"].keys()
    for field, value in [("guarded_merges", True), ("backend", "pdfium"), ("table_cell_matching", False)]:
        changed = config.model_copy(deep=True)
        setattr(changed.parsing, field, value)
        assert parse_signature(changed, "structured") != legacy_parse
    config.chunking.table_headers = "multirow"
    assert chunk_signature(config) != legacy_chunk
    assert parse_signature(config, "structured") == legacy_parse


def test_quality_flags_removed_table_number_and_bad_span(local):
    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    snapshot = load_docling(parsed).model_dump(mode="json")
    original = digest(snapshot)
    cell = next(c for c in snapshot["tables"][0]["data"]["table_cells"] if c["text"] == "150")
    cell["text"] = ""
    item = next(t for t in snapshot["texts"] if t.get("prov"))
    item["prov"][0]["charspan"] = [0, len(item["orig"]) + 2]
    modified = digest(snapshot)
    doc = store.meta("dataset")["documents"][0]
    quality = inspect_snapshot(snapshot, doc["pdf_path"])
    assert quality["counts"]["invalid_original_charspan"] == 1
    assert quality["counts"]["table_region_numbers_not_assigned"] >= 1
    assert digest(snapshot) == modified != original


def test_sample_success_cache_checksum_and_no_database_mutation(local, monkeypatch):
    from docling.datamodel.base_models import InputFormat

    from financial_rag import parsing

    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    doc = load_docling(parsed)
    calls = []

    def convert(*args, **kwargs):
        calls.append(kwargs)
        return NS(document=doc, pages=[], errors=[], status=NS(value="success"))

    converter = NS(format_to_options={InputFormat.PDF: NS(pipeline_options=NS())}, convert=convert)
    monkeypatch.setattr(parsing, "_converter", lambda _: converter)
    before = store.db.total_changes
    first = parse_sample(config, store, "fictional-bank", 1, 3)
    assert not first["failed"] and len(calls) == 1
    second = parse_sample(config, store, "fictional-bank", 1, 3)
    assert second["cached"] and len(calls) == 1
    from pathlib import Path

    Path(first["nodes"]).write_text("[]", encoding="utf-8")
    parse_sample(config, store, "fictional-bank", 1, 3)
    assert len(calls) == 2 and store.db.total_changes == before
