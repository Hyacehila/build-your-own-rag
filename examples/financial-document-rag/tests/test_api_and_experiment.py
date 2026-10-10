from __future__ import annotations

import json
import math

import pytest

from financial_rag.api import APIError, ModelAPI, embedding_key, usage_summary
from financial_rag.chunking import chunk, current_index
from financial_rag.common import RagError
from financial_rag.config import Role
from financial_rag.enrichment import enrich
from financial_rag.evaluation import evaluate, report, retrieval_metrics
from financial_rag.retrieval import Search, embed_index, page_ranking
from financial_rag.runner import run


def test_api_cache_reuse_and_invalidation(configured):
    config, store, fake = configured
    api = ModelAPI(config, store, "unit")
    vectors = api.embed(["same", "another", "same"])
    assert len(fake.requests) == 1
    assert (vectors[0] == vectors[2]).all()
    api.embed(["same"])
    assert len(fake.requests) == 1
    config.models.embedding.cache_revision = "2"
    api.embed(["same"])
    assert len(fake.requests) == 2
    api.embed(["changed"])
    assert len(fake.requests) == 3
    config.models.embedding.model = "different"
    api.embed(["same"])
    assert len(fake.requests) == 4
    assert usage_summary(store.calls("unit"))["embedding"]["requests"] == 4


def test_interface_failures_are_not_cached_and_usage_can_be_unknown(configured):
    config, store, fake = configured
    api = ModelAPI(config, store, "errors")
    fake.override = lambda path, body: (401, {"error": "DO NOT LOG SECRET response body"})
    with pytest.raises(APIError, match="HTTP 401"):
        api.embed(["test"])
    assert store.vector(embedding_key(config.models.embedding, "test")) is None
    assert "SECRET" not in json.dumps(store.calls("errors"))
    fake.override = lambda path, body: (200, {"data": [{"index": 0, "embedding": [0.0, 0.0]}]})
    with pytest.raises(APIError, match="nonzero"):
        api.embed(["test"])
    fake.override = lambda path, body: (200, {"data": [{"index": 1, "embedding": [1.0, 2.0]}]})
    with pytest.raises(APIError, match="indices"):
        api.embed(["test"])
    fake.override = None
    fake.usage = False
    api.embed(["test"])
    calls = store.calls("errors")
    assert calls[-1]["usage"] is None
    with pytest.raises(RagError, match="Configure"):
        config.models.chat = Role()
        api.chat("chat", [{"role": "user", "content": "test"}])


def prepare_all(config, store):
    result = enrich(config, store)
    assert not result["failed"]
    chunk(config, store, "enriched")
    for kind in ("flat", "structured", "enriched"):
        embed_index(config, store, kind)


def test_failures_keep_denominator_and_judge_pending(configured):
    config, store, fake = configured
    embed_index(config, store, "flat")
    fake.override = lambda path, body: (
        (500, {"error": "test"}) if path.endswith("/chat/completions") else None
    )
    result = run(config, store, "failed", ["A"], limit=2)
    assert result["failures"] == 2
    config.models.judge = Role()
    assert evaluate(config, store, "failed")["rows"] == 2
    assert evaluate(config, store, "failed")["pending"] == 2
    fake.override = None
    run(config, store, "failed", ["A"], limit=2, retry_errors=True)
    assert all(len(r["previous_attempts"]) == 1 for r in store.records("failed"))
    with pytest.raises(RagError, match="records changed"):
        report(config, store, "failed")


def test_retrieval_metrics_and_page_deduplication():
    sources = [
        {"doc_id": "d", "doc_version": "v", "page_index": 1, "page_number": 2},
        {"doc_id": "d", "doc_version": "v", "page_index": 2, "page_number": 3},
    ]
    ranking = page_ranking([{"sources": sources, "score": 1}, {"sources": sources[:1], "score": 0.5}])
    assert len(ranking) == 2
    gold = [{**sources[0], "grade": 2}, {**sources[1], "grade": 1}]
    assert retrieval_metrics(ranking, gold) == {"ndcg_at_10": 1, "recall_at_5": 1, "recall_at_10": 1}
    assert retrieval_metrics([], gold)["ndcg_at_10"] == 0
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert retrieval_metrics(ranking[::-1], gold)["ndcg_at_10"] == pytest.approx(expected)


def test_fts_statistics_are_isolated_and_search_escapes_syntax(configured):
    config, store, _ = configured
    embed_index(config, store, "flat")
    index = current_index(config, store, "flat")
    search = Search(config, store, index, ModelAPI(config, store, "fts"))
    before = [c["id"] for c in search.search('revenue OR "2024" * (NEAR)')]
    prepare_all(config, store)
    after = [c["id"] for c in search.search('revenue OR "2024" * (NEAR)')]
    assert before == after
