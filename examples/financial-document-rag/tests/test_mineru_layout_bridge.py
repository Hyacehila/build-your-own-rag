import json
import zipfile

from financial_rag.common import read_json, sha256, write_json
from financial_rag.mineru_bridge import html_table_data
from financial_rag.mineru_layout_bridge import convert_document, infer_header_rows
from financial_rag.mineru_packets import prepare_document


def test_header_inference_preserves_negative_values_and_multicolumn_years():
    data = html_table_data(
        '<table><tr><td>In millions</td><td colspan="2">2024</td></tr>'
        "<tr><td>Item</td><td>Assets</td><td>Liabilities</td></tr>"
        "<tr><td>Loss</td><td>(365)</td><td>17</td></tr></table>"
    )
    before = [(c.text, c.row_span, c.col_span) for c in data.table_cells]
    report = infer_header_rows(data)
    assert report["rows"] == [0, 1]
    assert all(c.column_header for c in data.table_cells if c.start_row_offset_idx < 2)
    assert [(c.text, c.row_span, c.col_span) for c in data.table_cells] == before
    assert not any(c.column_header for c in data.table_cells if c.start_row_offset_idx == 2)


def test_conversion_uses_page_local_tables_not_merged_output(local, tmp_path):
    _, store = local
    original = store.meta("dataset")["documents"][0]
    bundle = tmp_path / "cloud"
    manifest = prepare_document(
        original,
        {"doc_id": original["doc_id"], "pdf_sha256": original["sha256"], "pages": [{"page_number": 2}]},
        bundle,
        core_size=1,
        overlap=1,
    )
    results = []
    for entry in manifest["files"]:
        pages = []
        for p in entry["pages"]:
            html = (
                "<table><tr><th>Year</th><th>Amount</th></tr><tr><td>2024</td><td>%d</td></tr></table>"
                % p["page_number"]
            )
            body = {"type": "table_body", "bbox": [10, 30, 180, 80], "lines": [{"spans": [{"html": html}]}]}
            pages.append(
                {
                    "page_idx": p["uploaded_page_index"],
                    "page_size": p["page_size"],
                    "preproc_blocks": [{"type": "table", "bbox": [10, 30, 180, 80], "blocks": [body]}],
                    "para_blocks": [{"type": "table", "blocks": []}],
                    "discarded_blocks": [],
                }
            )
        path = bundle / (entry["data_id"] + ".zip")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("layout.json", json.dumps({"_version_name": "3.4.4", "pdf_info": pages}))
        results.append(
            {"data_id": entry["data_id"], "state": "done", "zip_file": str(path), "sha256": sha256(path)}
        )
    write_json(bundle / "results.json", {"complete": True, "results": results})
    dest = tmp_path / "converted"
    report = convert_document(bundle, original, dest, [1])
    document = read_json(dest / "document.json")
    assert report["tables"] == 2 and report["pages"] == 2
    assert [t["prov"][0]["page_no"] for t in document["tables"]] == [1, 3]
    assert [t["data"]["table_cells"][-1]["text"] for t in document["tables"]] == ["1", "3"]
    assert {s["page_index"] for n in read_json(dest / "nodes.json") for s in n["sources"]} == {0, 2}
