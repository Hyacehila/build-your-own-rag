import json

import httpx
import pymupdf
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DoclingDocument, ProvenanceItem, Size

from financial_rag.common import RagError, read_json
from financial_rag.diagnostics.structural_repairs import (
    native_grid_candidate,
    repair_headings,
    semantic_content,
)
from financial_rag.mineru_bridge import convert_content_list, html_table_data
from financial_rag.mineru_cloud import prepare_trial, submit_trial
from financial_rag.parsing import active_parses, docling_nodes, load_docling
from financial_rag.provenance import localize_fragment


def test_mineru_html_spans_numbers_and_original_page_mapping():
    source = '<table><tr><th rowspan="2">Metric</th><th colspan="2">USD million</th></tr><tr><th>2024</th><th>2023</th></tr><tr><td>Balance</td><td>(365)</td><td>1,020.50</td></tr></table>'
    data = html_table_data(source)
    assert (data.num_rows, data.num_cols) == (3, 3)
    assert data.grid[2][1].text == "(365)" and data.grid[2][2].text == "1,020.50"
    assert data.grid[0][0].row_span == 2
    mapping = {
        "doc_id": "sample",
        "source_page_index": 100,
        "uploaded_page_index": 0,
        "page_size": [600, 800],
    }
    items = [
        {
            "type": "table",
            "table_body": source,
            "table_caption": ["Original caption"],
            "page_idx": 0,
            "bbox": [100, 200, 900, 700],
        }
    ]
    doc, report = convert_content_list(items, mapping)
    roundtrip = DoclingDocument.model_validate(doc.model_dump(mode="json"))
    assert roundtrip.tables[0].prov[0].page_no == 101
    box = roundtrip.tables[0].prov[0].bbox
    assert (box.l, box.t, box.r, box.b) == (60, 160, 540, 560)
    nodes = docling_nodes(roundtrip, {"doc_id": "sample", "id": "original-v1", "pages": 150}, "parse-v1")
    assert {s["page_index"] for n in nodes for s in n["sources"]} == {100}
    assert report["cell_bbox_available"] is False and report["page_only_refs"]
    with pytest.raises(RagError, match="page index"):
        convert_content_list([{**items[0], "page_idx": 1}], mapping)
    with pytest.raises(RagError, match="bbox"):
        convert_content_list([{**items[0], "bbox": None}], mapping)
    with pytest.raises(RagError, match="flat MinerU"):
        convert_content_list("Markdown has no source geometry", mapping)


def test_fragment_sources_do_not_inherit_unread_pages_or_guess_repeated_text():
    sources = [{"doc_version": "d", "page_index": i, "page_number": i + 1} for i in (0, 1)]
    node = {
        "id": "n",
        "text": "alpha beta",
        "sources": sources,
        "raw": {
            "orig": "alpha beta",
            "prov": [{"page_no": 1, "charspan": [0, 5]}, {"page_no": 2, "charspan": [6, 10]}],
        },
    }
    chunk = {"text": "Heading\nbeta", "headings": ["Heading"], "kind": "text", "sources": sources}
    result = localize_fragment(chunk, [node])
    assert result["sources"] == [sources[1]] and result["node_sources"] == sources
    assert result["fragment_spans"][0]["orig_start"] == 6
    repeated = {
        **node,
        "text": "beta beta",
        "raw": {
            "orig": "beta beta",
            "prov": [{"page_no": 1, "charspan": [0, 4]}, {"page_no": 2, "charspan": [5, 9]}],
        },
    }
    assert localize_fragment(chunk, [repeated])["source_alignment"] == "not_uniquely_aligned"
    assert chunk["sources"] == sources


def test_local_heading_repairs_preserve_text_and_stop_toc_and_exhibit_leakage(tmp_path):
    path = tmp_path / "headings.pdf"
    titles = [
        ("FORM 10-K CROSS-REFERENCE INDEX", 1, 1),
        ("Part IV", 1, 1),
        ("15. Exhibit and Financial Statement Schedules", 3, 1),
        ("Exhibit 13", 6, 2),
        ("Note 24: Earnings", 6, 2),
        ("Table 24.1: EPS", 6, 2),
    ]
    doc = DoclingDocument(name="fixture")
    with pymupdf.open() as pdf:
        for page_number in (1, 2):
            page = pdf.new_page(width=600, height=800)
            doc.add_page(page_no=page_number, size=Size(width=600, height=800))
            for i, (text, level, p) in enumerate(titles):
                if p != page_number:
                    continue
                y = 45 + i * 35
                page.insert_text((30, y), text, fontsize=11)
                doc.add_heading(
                    text=text,
                    level=level,
                    prov=ProvenanceItem(
                        page_no=p,
                        charspan=(0, len(text)),
                        bbox=BoundingBox(l=30, t=y - 14, r=560, b=y + 3, coord_origin=CoordOrigin.TOPLEFT),
                    ),
                )
        pdf.save(path)
    before = doc.model_dump(mode="json")
    repaired, report = repair_headings(doc, path)
    assert report["toc_pages"] == [1]
    assert doc.model_dump(mode="json") == before
    assert semantic_content(doc) == semantic_content(repaired)
    assert repaired.texts[-1].parent.cref == repaired.texts[-2].self_ref
    assert repaired.texts[-2].parent.cref == repaired.texts[-3].self_ref
    assert repaired.texts[-3].parent.cref == "#/body"


def test_native_grid_recovers_missing_cell_but_rejects_conflicting_number(local):
    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    doc = load_docling(parsed)
    pdf = store.meta("dataset")["documents"][0]["pdf_path"]
    table = doc.tables[0].model_copy(deep=True)
    cell = next(c for c in table.data.table_cells if c.text == "150")
    cell.text = ""
    fixed, report = native_grid_candidate(table, pdf)
    assert report["accepted"] and any(c.text == "150" for c in fixed.data.table_cells)
    assert cell.text == ""
    cell.text = "999999"
    _, report = native_grid_candidate(table, pdf)
    assert not report["accepted"]


def test_mineru_submission_budget_secret_isolation_and_no_automatic_retry(local, tmp_path, monkeypatch):
    _, store = local
    token = "offline-private-token"
    monkeypatch.setenv("MINERU_API_KEY", token)
    with pytest.raises(RagError, match="1-8"):
        prepare_trial(store, [("fictional-bank", i) for i in range(1, 10)], tmp_path / "over")
    bundle = tmp_path / "trial"
    prepare_trial(store, [("fictional-bank", 1)], bundle)
    seen = []

    def respond(request):
        seen.append(request)
        if request.url.host == "mineru.net":
            assert request.headers["authorization"] == "Bearer " + token
            assert len(json.loads(request.content)["files"]) == 1
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "offline-batch",
                        "file_urls": ["https://upload.example.org/one.pdf"],
                    },
                },
            )
        assert "authorization" not in request.headers
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = submit_trial(bundle, client)
        assert result["reserved_pages"] == 1 and result["post_attempts"] == 1
        with pytest.raises(RagError, match="already"):
            submit_trial(bundle, client)
    assert len(seen) == 2
    assert token not in (bundle / "job.json").read_text()
    failed = tmp_path / "failed"
    prepare_trial(store, [("fictional-bank", 1)], failed)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"code": "A0202", "msg": token}))
    ) as client:
        with pytest.raises(RagError, match="A0202"):
            submit_trial(failed, client)
    assert token not in (failed / "job.json").read_text()
    assert read_json(failed / "job.json")["state"] == "stopped"
