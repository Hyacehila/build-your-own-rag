from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_rag.chunking import current_index
from financial_rag.common import RagError, read_jsonl, sha256
from financial_rag.dataset import download
from financial_rag.enrichment import enrich, enrichment_key
from financial_rag.parsing import active_parses, load_docling, parse


def test_parse_retry_removes_stale_partial_nodes(local):
    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    original = store.nodes(parsed["id"])
    assert len(original) > 1
    # A partial retry must still be a self-consistent tree, not dangling children.
    partial = [{**original[0], "children": [], "related_refs": []}]
    store.put_parse(parsed, partial)
    assert [n["id"] for n in store.nodes(parsed["id"])] == [original[0]["id"]]


def test_embedding_estimate_works_before_structured_index(local):
    from financial_rag.retrieval import estimate

    config, store = local
    with store.db:
        store.db.execute("DELETE FROM metadata WHERE key='index:structured'")
    result = estimate(config, store)
    assert result["indices"][0]["incremental_tokens"] > 0
    assert result["indices"][1]["status"] == "not_chunked"
    assert result["indices"][2]["status"] == "not_chunked"


def test_verified_model_bundle_rejects_changed_weights(tmp_path):
    from financial_rag.common import write_json
    from financial_rag.models import verify_models

    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"offline model integrity fixture")
    write_json(tmp_path / "manifest.json", {"files": [{"path": weight.name, "sha256": sha256(weight)}]})
    assert verify_models(tmp_path)
    weight.write_bytes(b"changed weights")
    with pytest.raises(RagError, match="changed or are incomplete"):
        verify_models(tmp_path)


def test_hub_download_directory_handles_windows_long_paths(tmp_path):
    from financial_rag.dataset import hub_local_dir

    path = hub_local_dir(tmp_path / ("nested" * 12) / ("cache" * 10))
    path.mkdir(parents=True)
    target = path / ("a" * 140 + ".incomplete")
    try:
        target.write_bytes(b"download")
        assert target.read_bytes() == b"download"
    finally:
        target.unlink(missing_ok=True)
        path.rmdir()
        path.parent.rmdir()


def test_document_enrichment_never_opens_questions_or_labels(configured, monkeypatch):
    config, store, fake = configured
    original = Path.open

    def document_only(path, *args, **kwargs):
        if path.name in ("queries.jsonl", "labels.jsonl"):
            raise AssertionError("Document enrichment must not read QA or labels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", document_only)
    result = enrich(config, store)
    assert not result["failed"]
    assert not any("reference_answer" in json.dumps(body) for _, body in fake.requests)
    index = current_index(config, store, "structured")
    asset = next(n for n in store.nodes(index["parse_ids"][0]) if n["kind"] == "picture")
    old = enrichment_key(config, asset)
    config.models.vision.cache_revision = "new-weights"
    assert enrichment_key(config, asset) != old
    assert store.cached(enrichment_key(config, asset)) is None


def test_empty_table_region_uses_original_image_and_vision_identity(configured):
    config, store, fake = configured
    parsed = active_parses(config, store, "structured")[0]
    nodes = store.nodes(parsed["id"])
    table = next(n for n in nodes if n["kind"] == "table")
    table["text"] = ""
    store.put_parse(parsed, nodes)
    result = enrich(config, store)
    assert not result["failed"]
    assert result["empty_table_image_fallbacks"] == [table["id"]]
    key = enrichment_key(config, table)
    record = store.cached(key)
    assert record["input_mode"] == "empty_table_image_fallback"
    request = next(body for _, body in fake.requests if "extracted no cells" in json.dumps(body))
    assert request["model"] == config.models.vision.model
    assert any(p["type"] == "image_url" for p in request["messages"][0]["content"])
    assert record["sources"] == table["sources"]
    config.models.vision.cache_revision = "new-vision"
    assert enrichment_key(config, table) != key


def test_parser_does_not_mistake_fixture_snapshot_for_real_conversion(local, monkeypatch):
    config, store = local
    import financial_rag.parsing as module

    snapshot = active_parses(config, store, "structured")[0]
    document = load_docling(snapshot)
    calls = []

    class Converter:
        format_to_options = {
            "pdf": SimpleNamespace(pipeline_options=SimpleNamespace(model_dump=lambda **_: {}))
        }

        def convert(self, path, **kwargs):
            calls.append(path)
            return SimpleNamespace(document=document, status=SimpleNamespace(value="success"), errors=[])

    monkeypatch.setattr(module, "_converter", lambda _: Converter())
    assert parse(config, store, "structured")["completed"] == ["fictional-bank"]
    assert len(calls) == 1
    assert parse(config, store, "structured")["cached"] == ["fictional-bank"]
    assert len(calls) == 1
    # A corrupted snapshot invalidates the cache, even with unchanged configuration.
    new = active_parses(config, store, "structured")[0]
    Path(new["snapshot"]).write_text("{}", encoding="utf-8")
    assert parse(config, store, "structured")["completed"] == ["fictional-bank"]
    assert len(calls) == 2


def test_parse_failure_is_persisted_and_blocks_index(local, monkeypatch):
    config, store = local
    import financial_rag.parsing as module

    def unavailable(_):
        raise RuntimeError("test layout model unavailable")

    monkeypatch.setattr(module, "_converter", unavailable)
    result = parse(config, store, "structured")
    assert result["failed"] == ["fictional-bank"]
    with pytest.raises(RagError, match="Incomplete"):
        active_parses(config, store, "structured")


def test_download_validates_upstream_hashes_and_joins_full_mapping(local, monkeypatch):
    import huggingface_hub

    import financial_rag.dataset as module

    config, store = local
    original = store.meta("dataset")
    doc = original["documents"][0]
    dest = config.work_dir / "raw" / config.dataset.revision
    (dest / "pdfs").mkdir(parents=True)
    pdf = dest / "pdfs" / doc["filename"]
    pdf.write_bytes(Path(doc["pdf_path"]).read_bytes())

    def parquet(name, rows):
        target = dest / name / "test-00000-of-00001.parquet"
        target.parent.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), target)

    queries = read_jsonl(Path(original["prepared_dir"]) / "queries.jsonl")
    labels = read_jsonl(Path(original["prepared_dir"]) / "labels.jsonl")
    parquet(
        "queries",
        [
            {**q, **{k: v for k, v in label.items() if k not in ("qrels", "query_id")}, "language": "english"}
            for q, label in zip(queries, labels)
        ]
        + [{"query_id": "discarded", "query": "DROP_ME_NON_ENGLISH", "language": "unsupported"}],
    )
    parquet("documents_metadata", [{"file_name": doc["filename"], "doc_id": doc["doc_id"], "page_number": 3}])
    parquet(
        "corpus", [{"doc_id": doc["doc_id"], "corpus_id": str(i), "page_number_in_doc": i} for i in range(3)]
    )
    parquet(
        "qrels",
        [
            {"query_id": label["query_id"], "corpus_id": str(q["page_index"]), "score": q["grade"]}
            for label in labels
            for q in label["qrels"]
        ],
    )
    siblings = [
        SimpleNamespace(
            rfilename=p.relative_to(dest).as_posix(),
            size=p.stat().st_size,
            lfs={"sha256": sha256(p)},
            blob_id=None,
        )
        for p in dest.rglob("*")
        if p.is_file()
    ]
    info = SimpleNamespace(sha=config.dataset.revision, siblings=siblings)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *a, **k: info))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(dest))

    def annotation_download(repo, filename, **kwargs):
        target = Path(kwargs["local_dir"]) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((dest / filename).read_bytes())
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", annotation_download)
    monkeypatch.setattr(module, "PDF_PAGES", {doc["filename"]: 3})
    config.dataset.expected_documents = 1
    config.dataset.expected_pages = 3
    config.dataset.expected_english_queries = 3
    result = download(config, store)
    assert result["counts"] == {"documents": 1, "pages": 3, "queries": 3}
    assert result["files"]["pdfs/" + doc["filename"]] == sha256(pdf)
    assert result["language"] == "english"
    assert not list(dest.glob("queries/*.parquet"))
    assert not list(dest.glob("qrels/*.parquet"))
    assert not list(config.work_dir.glob("finance-english-*"))
    assert "DROP_ME_NON_ENGLISH" not in (Path(result["prepared_dir"]) / "queries.jsonl").read_text(
        encoding="utf-8"
    )
    with pdf.open("r+b") as f:
        f.seek(-8, 2)
        f.write(b"BAD")
    with pytest.raises(RagError, match="SHA-256 mismatch"):
        download(config, store)
