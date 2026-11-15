import io
import json
import zipfile

import httpx
import pymupdf
import pytest
from clean_documents.common import RagError
from clean_documents.mineru_cloud import fetch_trial
from clean_documents.mineru_packets import prepare_document, submit_document


def test_packet_mapping_and_reserved_submission_without_network(tmp_path, monkeypatch):
    pdf_path = tmp_path / "source.pdf"
    with pymupdf.open() as pdf:
        for _ in range(3):
            pdf.new_page()
        pdf.save(pdf_path)
    bundle = tmp_path / "bundle"
    doc = {"id": "doc", "doc_id": "doc", "pdf_path": str(pdf_path), "pages": 3}
    with pytest.raises(RagError, match="exceed the declared maximum"):
        prepare_document(doc, {"doc_id": "doc", "pages": []}, bundle,
                         selected_pages=[1, 3], language="en", max_pages=1)
    assert not bundle.exists()
    manifest = prepare_document(doc, {"doc_id": "doc", "pages": []}, bundle,
                                selected_pages=[1, 3], language="en", max_pages=3)
    entry = manifest["files"][0]
    assert [page["page_number"] for page in entry["pages"]] == [1, 3]
    assert "sha256" not in entry
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("layout.json", json.dumps({"_version_name": "3.4.4", "pdf_info": []}))
    captured = []
    def handler(request):
        captured.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"code": 0, "data": {
                "batch_id": "batch", "file_urls": ["https://upload.invalid/file"]}})
        if request.method == "PUT":
            return httpx.Response(200)
        if request.url.host == "mineru.net":
            return httpx.Response(200, json={"code": 0, "data": {"extract_result": [{
                "file_name": entry["filename"], "data_id": entry["data_id"], "state": "done",
                "full_zip_url": "https://download.invalid/result"}]}})
        return httpx.Response(200, content=blob.getvalue())
    monkeypatch.setenv("MINERU_API_KEY", "test-only")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert submit_document(bundle, client)["state"] == "submitted"
        with pytest.raises(RagError, match="already reserved"):
            submit_document(bundle, client)
        assert fetch_trial(bundle, client)["complete"]
    assert "authorization" not in next(r for r in captured if r.method == "PUT").headers
