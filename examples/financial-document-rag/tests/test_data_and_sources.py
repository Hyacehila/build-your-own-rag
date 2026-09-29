from __future__ import annotations

import copy

import pytest

from financial_rag.chunking import current_index, flat_chunks
from financial_rag.common import RagError
from financial_rag.config import Chunking
from financial_rag.dataset import inspect, prepare_annotations
from financial_rag.sources import Reader, validate_citations
from financial_rag.tokenization import Tokenizer, recursive_spans


def test_english_filter_and_page_origin_are_validated_over_all_rows():
    documents = [{"doc_id": "d", "id": "v", "pages": 3}]
    corpus = [{"doc_id": "d", "corpus_id": str(p), "page_number_in_doc": p} for p in (1, 2, 3)]
    queries = [
        {"query_id": str(i), "query": f"q{i}", "language": lang}
        for i, lang in enumerate(["english", "unsupported", "unsupported", "english"])
    ]
    qrels = [{"query_id": "0", "corpus_id": "3", "score": 2}, {"query_id": "3", "corpus_id": "1", "score": 1}]
    runtime, labels, audit = prepare_annotations(queries, qrels, corpus, documents, 2)
    assert [q["query_id"] for q in runtime] == ["0", "3"]
    assert audit["page_origins"] == {"d": 1}
    assert audit["language"] == "english"
    assert "languages" not in audit and "all_query_rows" not in audit
    assert labels[0]["qrels"][0]["page_index"] == 2
    assert all(set(q) == {"query_id", "query"} for q in runtime)
    with pytest.raises(RagError, match="page mapping"):
        prepare_annotations(queries, qrels, corpus[:-1], documents, 2)
    with pytest.raises(RagError, match="English"):
        prepare_annotations(queries, qrels, corpus, documents, 3)
    wrong = copy.deepcopy(qrels)
    wrong[0]["corpus_id"] = "99"
    with pytest.raises(RagError, match="unknown corpus"):
        prepare_annotations(queries, wrong, corpus, documents, 2)


def test_recursive_offsets_overlap_unicode_and_cross_page(local):
    config, store = local
    tokenizer = Tokenizer(config.tokenizer)
    text = "标题 😀\n\n" + "Revenue rose 25 percent.\n" * 50
    spans = recursive_spans(text, tokenizer, 55, 8)
    covered = set()
    for a, b in spans:
        assert tokenizer.count(text[a:b]) <= 55
        covered.update(range(a, b))
    assert covered == set(range(len(text)))
    assert any(a < spans[i - 1][1] for i, (a, _) in enumerate(spans) if i)
    index = current_index(config, store, "flat")
    nodes = store.nodes(index["parse_ids"][0])
    config.chunking = Chunking(max_tokens=5000, overlap=100)
    chunks = flat_chunks(nodes, config, tokenizer)
    assert any(len({s["page_index"] for s in c["sources"]}) > 1 for c in chunks)
    assert all(s["page_number"] == s["page_index"] + 1 for c in chunks for s in c["sources"])


def test_blank_pages_do_not_inherit_adjacent_chunk_hits(local):
    config, store = local
    index = current_index(config, store, "flat")
    nodes = copy.deepcopy(store.nodes(index["parse_ids"][0]))
    nodes[1]["text"] = "\n  \n"
    config.chunking = Chunking(max_tokens=5000, overlap=100)
    pieces = flat_chunks(nodes, config, Tokenizer(config.tokenizer))
    pages = {s["page_index"] for c in pieces for s in c["sources"]}
    assert pages == {0, 2}
    assert any(len(c["sources"]) == 2 for c in pieces)


def test_hybrid_heading_path_table_headers_and_original_read(local):
    config, store = local
    index = current_index(config, store, "structured")
    chunks = store.chunks(index["id"])
    assert any(c["kind"] == "picture" and c["asset_ids"] for c in chunks)
    tables = [c for c in chunks if c["kind"] == "table"]
    assert len(tables) > 1
    assert all("2023 USD m" in c["text"] and "2024 USD m" in c["text"] for c in tables)
    assert all(c["headings"] == ["1. Performance", "1.1 Revenue by business unit"] for c in tables)
    reader = Reader(config, store, index)
    tail = next(c for c in tables if "Test segment 24" in c["text"])
    result = reader.read(chunk_id=tail["id"])
    assert "error" not in result
    assert any("Test segment 24" in e.get("text", "") for e in reader.evidence.values())
    assert reader.images
    check = validate_citations(["E1", "invented"], reader.public_evidence())
    assert check["validity"] == 0.5
    assert check["bindings"]["E1"][0]["doc_version"]
    assert not validate_citations([], reader.public_evidence())["validity"]


def test_inspection_and_read_budget_and_version_binding(local):
    config, store = local
    report = inspect(config, store, tables=True)
    assert report["documents"] == 1 and report["failed_pages"] == 0
    index = current_index(config, store, "structured")
    config.query.text_tokens = 40
    config.query.page_images = 1
    reader = Reader(config, store, index)
    for c in store.chunks(index["id"]):
        reader.preview(c)
        reader.read(chunk_id=c["id"])
    assert reader.text_used <= 40
    assert len(reader.images) <= 1
    flat = current_index(config, store, "flat")
    response = reader.read(chunk_id=store.chunks(flat["id"])[0]["id"])
    assert "different index" in response["error"]
    version = next(iter(reader.allowed_versions))
    assert "error" in reader.read(doc_version=version, page_index=100)
    doc = store.get("documents", version)
    from pathlib import Path

    with Path(doc["pdf_path"]).open("ab") as f:
        f.write(b"tampered")
    fresh = Reader(config, store, index)
    assert "modified" in fresh.read(doc_version=version, page_index=0)["error"]


def test_cache_invalidation_after_chunk_settings(local):
    config, store = local
    config.chunking.max_tokens += 1
    with pytest.raises(RagError, match="stale"):
        current_index(config, store, "flat")
