from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from financial_rag.api import APIError, ModelAPI, embedding_key, usage_summary
from financial_rag.chunking import chunk, current_index
from financial_rag.common import RagError, read_jsonl
from financial_rag.config import Role
from financial_rag.enrichment import enrich
from financial_rag.evaluation import evaluate, report, retrieval_metrics
from financial_rag.retrieval import Search, embed_index, estimate, page_ranking
from financial_rag.runner import QuestionRunner, run


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


def test_four_arms_local_http_end_to_end_and_resume(configured):
    config, store, fake = configured
    before = estimate(config, store)
    assert before["projected_assets_separate"]["table_count"] == 1
    assert before["projected_assets_separate"]["picture_count"] == 1
    prepare_all(config, store)
    after = estimate(config, store)
    assert after["measured_text_estimated_tokens_to_embed"] == 0
    assert after["estimated_embedding_cost_excluding_projected_assets"] == 0
    assert enrich(config, store)["cached"] == 2
    result = run(config, store, "offline-test", ["A", "B", "C", "D"])
    assert result["records"] == 12 and result["failures"] == 0
    assert result["synthetic"] and not result["full_benchmark"]
    records = store.records("offline-test")
    by_key = {(r["arm"], r["query_id"]): r for r in records}
    for r in records:
        assert r["text_tokens_read"] <= config.query.text_tokens
        assert r["page_images_read"] <= config.query.page_images
        assert r["citation_check"]["validity"] == 1
        if r["arm"] == "D":
            other = by_key[("C", r["query_id"])]
            assert r["index_id"] == other["index_id"]
            assert r["initial_page_ranking"] == other["initial_page_ranking"]
            assert r["model_calls"] <= 6 and r["tool_calls"] <= 8
    calls_before_resume = len(fake.requests)
    run(config, store, "offline-test", ["A", "B", "C", "D"])
    assert len(fake.requests) == calls_before_resume
    config.models.judge = Role()
    assert evaluate(config, store, "offline-test")["pending"] == 12
    output = report(config, store, "offline-test")
    assert "NOT BENCHMARK RESULTS" in Path(output["report"]).read_text(encoding="utf-8")
    config.models.judge = Role(base_url=fake.url, model="synthetic-judge", retries=0)
    assert evaluate(config, store, "offline-test")["pending"] == 0
    scored = read_jsonl(config.work_dir / "runs/offline-test/scored.jsonl")
    assert all(r["judge"]["status"] == "scored" for r in scored)
    assert any(
        "image_url" in json.dumps(body) for path, body in fake.requests if path.endswith("/chat/completions")
    )
    report(config, store, "offline-test")
    assert (config.work_dir / "runs/offline-test/deltas.csv").exists()


def test_agent_hard_loop_caps_and_final_without_tools(configured):
    config, store, fake = configured
    prepare_all(config, store)
    index = current_index(config, store, "enriched")

    calls_per_batch = [6]

    def endless(path, body):
        if "tools" in body:
            return 200, {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{i}",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"query":"cash"}'},
                                }
                                for i in range(calls_per_batch[0])
                            ],
                        }
                    }
                ]
            }
        return None

    fake.override = endless
    config.query.agent_model_calls = 6
    config.query.agent_tool_calls = 8
    worker = QuestionRunner(config, store, index, ModelAPI(config, store, "caps"))
    result = worker.run({"query_id": "limit", "query": "cash"}, "D")
    assert result["status"] == "ok"
    assert result["tool_calls"] == 8
    assert result["model_calls"] <= 6
    assert any(t["tool"] == "rejected" for t in result["trace"])
    chat_bodies = [b for p, b in fake.requests if p.endswith("/chat/completions")]
    assert "tools" not in chat_bodies[-1]
    calls_per_batch[0] = 1
    worker = QuestionRunner(config, store, index, ModelAPI(config, store, "model-cap"))
    result = worker.run({"query_id": "limit", "query": "cash"}, "D")
    assert result["model_calls"] == 6
    assert result["tool_calls"] == 8  # search + two seed reads + five planning calls
    assert result["status"] == "ok"
    config.query.agent_model_calls = 1
    worker = QuestionRunner(config, store, index, ModelAPI(config, store, "one-call"))
    result = worker.run({"query_id": "limit", "query": "cash"}, "D")
    assert result["model_calls"] == 1


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
