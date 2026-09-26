import pytest
from docling_core.types.doc import DocItemLabel
from test_cleaning import final_outline_receipt

from financial_rag.api import ModelAPI
from financial_rag.chunking import chunk
from financial_rag.cleaning import build, export_release
from financial_rag.common import RagError, write_json
from financial_rag.config import Chunking, Role, Tokens
from financial_rag.dataset import dataset
from financial_rag.final_outline import accept
from financial_rag.parsing import active_parses, load_docling, parse
from financial_rag.retrieval import Search, embed_index
from financial_rag.retrieval_release import check, export, prepare, reuse_embeddings
from financial_rag.storage import Store


def accepted_fixture(config, source, tmp_path):
    doc = source.meta("dataset")["documents"][0]
    raw = source.get("parses", source.meta("parse:structured:" + doc["id"]))
    snapshot = load_docling(raw)
    workspace, profiles, output = (tmp_path / name for name in ("prep", "profiles", "facts"))
    write_json(workspace / "sources.json", [{"document": doc, "raw": raw}])
    write_json(workspace / "diagnostics" / (doc["doc_id"] + ".json"), {"issues": []})
    write_json(
        profiles / (doc["doc_id"] + ".json"),
        {
            "doc_id": doc["doc_id"],
            "pdf_sha256": doc["sha256"],
            "raw_snapshot_sha256": raw["snapshot_sha256"],
            "excluded_pages": [],
            "boundaries": [1],
            "reviewed_pages": [1],
            "unresolved": [],
            "heading_levels": {
                h.self_ref: h.level for h in snapshot.texts if h.label == DocItemLabel.SECTION_HEADER
            },
        },
    )
    release = build(workspace, profiles, output, config)
    with Store(output) as store, pytest.raises(RagError, match="semantic acceptance"):
        active_parses(config, store, "structured")
    _, receipt = final_outline_receipt(output, release)
    write_json(tmp_path / "reviews" / doc["doc_id"] / "receipt.json", receipt)
    accept(output, tmp_path / "reviews")
    database = tmp_path / "facts.sqlite"
    export_release(output, database)
    return database, output / "raw"


def test_accepted_facts_to_retrieval_and_portable_restore(local, fake, tmp_path):
    original_config, source = local
    database, pdf_dir = accepted_fixture(original_config, source, tmp_path)
    config = original_config.model_copy(deep=True)
    config.work_dir = tmp_path / "retrieval"
    config.tokenizer = Tokens()
    config.chunking = Chunking(max_tokens=800, overlap=100, table_headers="multirow", source_scope="fragment")
    config.models.embedding = Role(
        base_url=fake.url, model="offline-double", request_params={"dimensions": 32}
    )
    prepare(config, database, pdf_dir)
    with Store(config.work_dir) as store:
        assert dataset(store)["document_only"]
        assert dataset(store)["synthetic"]  # Never advertise a fictional release as real evidence.
        with pytest.raises(RagError, match="document-only"):
            dataset(store, verify_queries=True)
        assert parse(config, store, "structured")["completed"] == []
        doc = dataset(store)["documents"][0]
        pointer = "parse:structured:" + doc["id"]
        accepted_id = store.meta(pointer)
        store.set_meta(pointer, store.meta("rawparse:structured:" + doc["id"]))
        with pytest.raises(RagError, match="accepted fact release"):
            active_parses(config, store, "structured")
        store.set_meta(pointer, accepted_id)
        parse(config, store, "flat")
        for kind in ("flat", "structured"):
            chunk(config, store, kind)
        check(config, store, require_vectors=False)
        with pytest.raises(RagError, match="embedding|vector"):
            check(config, store)
        for kind in ("flat", "structured"):
            embed_index(config, store, kind)
        before = store.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        index = chunk(config, store, "structured")
        assert store.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == before
        assert Search(config, store, index, ModelAPI(config, store, "offline-check")).search("revenue")
        report = check(config, store)
        assert report["status"] == "passed"
        portable = tmp_path / "retrieval.sqlite"
        export(config, store, portable)
    restored_config = config.model_copy(deep=True)
    restored_config.work_dir = tmp_path / "restored"
    prepare(restored_config, portable, pdf_dir)
    with Store(restored_config.work_dir) as restored:
        assert check(restored_config, restored)["status"] == "passed"
        assert restored.meta("retrieval_release")["api_usage"]  # Export keeps aggregate build usage.
        assert restored.meta("retrieval_release")["local_api_usage"] == {}
        count = len(fake.requests)
        for kind in ("flat", "structured"):
            embed_index(restored_config, restored, kind)
        assert len(fake.requests) == count
        with restored.db:
            restored.db.execute("DELETE FROM chunk_vectors")
            restored.db.execute("DELETE FROM embeddings")
        reuse = reuse_embeddings(restored_config, restored, portable)
        assert reuse["copied"] > 0
        assert check(restored_config, restored)["status"] == "passed"
        restored_config.models.embedding.cache_revision = "changed"
        assert reuse_embeddings(restored_config, restored, portable)["copied"] == 0
        with pytest.raises(RagError, match="embedding|vector"):
            check(restored_config, restored)

    # Same key/model but malformed dimensions must never enter the new cache.
    import sqlite3

    with sqlite3.connect(portable) as incoming:
        incoming.execute("UPDATE embeddings SET dimensions=31")
    restored_config.models.embedding.cache_revision = "1"
    with Store(restored_config.work_dir) as restored, pytest.raises(RagError, match="dimensions"):
        reuse_embeddings(restored_config, restored, portable)


def test_hybrid_uses_fact_parents_at_independent_appendix_boundary(tmp_path):
    from docling_core.types.doc import DoclingDocument, Size
    from test_page_exclusions import provenance

    from financial_rag.chunking import hybrid_chunks
    from financial_rag.config import Config
    from financial_rag.parsing import docling_nodes
    from financial_rag.tokenization import Tokenizer

    doc = DoclingDocument(name="two independent scopes")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    first = doc.add_heading(text="Previous exhibit", level=1, prov=provenance(1, "Previous exhibit"))
    doc.add_text(label=DocItemLabel.TEXT, text="Old body", parent=first, prov=provenance(1, "Old body"))
    # Level 2 does not imply a child: an independently reviewed scope owns this heading.
    group = doc.add_group(name="source-range:appendix")
    second = doc.add_heading(text="New appendix", level=2, parent=group, prov=provenance(1, "New appendix"))
    doc.add_text(
        label=DocItemLabel.TEXT, text="New body " * 900, parent=second, prov=provenance(1, "New body " * 900)
    )
    path = tmp_path / "doc.json"
    write_json(path, doc.model_dump(mode="json"))
    nodes = docling_nodes(doc, {"doc_id": "scope", "id": "version", "pages": 1}, "parse")
    config = Config(chunking=Chunking(max_tokens=100, overlap=10, table_headers="multirow"))
    tokenizer = Tokenizer(config.tokenizer)
    chunks = hybrid_chunks({"snapshot": str(path)}, nodes, config, tokenizer)
    current = [c for c in chunks if "New body" in c["text"]]
    assert len(current) > 1
    assert all(c["headings"] == ["New appendix"] and "Previous exhibit" not in c["text"] for c in current)
    assert all(tokenizer.count(c["text"]) <= 100 for c in chunks)
