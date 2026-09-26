import json

import httpx
import pytest

from financial_rag.common import RagError, read_json
from financial_rag.mineru_packets import prepare_document, submit_document


def test_document_trial_maps_excluded_pages_and_reserves_once(local, tmp_path, monkeypatch):
    _, store = local
    doc = store.meta("dataset")["documents"][0]
    annotation = {"doc_id": doc["doc_id"], "pdf_sha256": doc["sha256"], "pages": [{"page_number": 2}]}
    bundle = tmp_path / "comparison"
    manifest = prepare_document(doc, annotation, bundle, core_size=1, overlap=1)
    owners = [p["page_number"] for f in manifest["files"] for p in f["pages"] if p["owner"]]
    assert owners == [1, 3]
    assert manifest["submitted_pages"] == 4
    token = "test-private-token"
    monkeypatch.setenv("MINERU_API_KEY", token)
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "mineru.net":
            assert request.headers["authorization"] == "Bearer " + token
            payload = json.loads(request.content)
            assert payload["model_version"] == "vlm" and payload["enable_table"]
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "offline-batch",
                        "file_urls": [f"https://upload.test/{i}" for i in range(2)],
                    },
                },
            )
        assert "authorization" not in request.headers
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        job = submit_document(bundle, client=client)
        assert job["reserved_pages"] == 4 and job["post_attempts"] == 1
        with pytest.raises(RagError, match="already reserved"):
            submit_document(bundle, client=client)
    assert len(requests) == 3
    assert token not in (bundle / "job.json").read_text()
    assert read_json(bundle / "job.json")["state"] == "submitted"


def test_document_trial_rejects_changed_source_and_oversized_packets(local, tmp_path):
    _, store = local
    doc = store.meta("dataset")["documents"][0]
    annotation = {"doc_id": doc["doc_id"], "pdf_sha256": "wrong", "pages": []}
    with pytest.raises(RagError, match="another PDF"):
        prepare_document(doc, annotation, tmp_path)
    with pytest.raises(RagError, match="200-page"):
        prepare_document(doc, annotation, tmp_path, core_size=200, overlap=2)
