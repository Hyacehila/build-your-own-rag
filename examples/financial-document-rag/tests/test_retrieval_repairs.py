"""Regressions for lost anchors, additive assets, bounded recovery and fair evidence budgets."""

import json

import pytest

from financial_rag.api import APIError, ModelAPI
from financial_rag.chunking import chunk, current_index
from financial_rag.common import RagError
from financial_rag.enrichment import _describe, enrich, enrichment_key
from financial_rag.evaluation import summarize
from financial_rag.retrieval import Search, embed_index
from financial_rag.runner import QuestionRunner
from financial_rag.sources import Reader


@pytest.mark.parametrize("failure", ["length", "empty"])
def test_descriptions_retry_incomplete_output_without_forcing_answer_json(configured, monkeypatch, failure):
    config, store, fake = configured
    monkeypatch.setattr("financial_rag.api.time.sleep", lambda _: None)
    config.models.chat.json_mode = True
    count = 0

    def response(path, body):
        nonlocal count
        count += 1
        assert "response_format" not in body
        return 200, {
            "choices": [
                {
                    "finish_reason": "length" if failure == "length" and count == 1 else "stop",
                    "message": {
                        "role": "assistant",
                        "content": "" if count == 1 else "Original chart retrieval description",
                    },
                }
            ]
        }

    fake.override = response
    text = _describe(
        ModelAPI(config, store, "description-retry"), "chat", [{"role": "user", "content": "Describe source"}]
    )
    assert count == 2 and text == "Original chart retrieval description"
    assert [c["validation_attempt"] for c in store.calls("description-retry")] == [1, 2]


def test_legacy_scored_rows_do_not_invent_new_search_metrics():
    row = {
        "arm": "A",
        "groups": {},
        "status": "ok",
        "judge": {"correct": True, "status": "scored"},
        "citation_check": {"total": 1},
        "metrics": {
            "ndcg_at_10": 1,
            "recall_at_5": 1,
            "recall_at_10": 1,
            "evidence_page_coverage": 1,
            "citation_validity": 1,
        },
        "elapsed_seconds": 1,
        "model_calls": 1,
        "tool_calls": 1,
    }
    result = summarize([row])[0]
    assert result["search_union_recall"] is None
    assert result["search_union_recall_n"] == 0


def test_tail_rows_survive_small_budget_and_previews_do_not_spend_evidence(local):
    config, store = local
    index = current_index(config, store, "structured")
    target = next(c for c in store.chunks(index["id"]) if "Test segment 24" in c["text"])
    reader = Reader(config, store, index)
    config.query.text_tokens = reader.tokenizer.count(target["text"])
    config.query.read_max_tokens = config.query.text_tokens
    for c in store.chunks(index["id"]):
        reader.preview(c)
    assert reader.text_used == 0
    assert reader.preview_used <= config.query.preview_total_tokens
    reader.read(chunk_id=target["id"], include_images=False)
    assert any("Test segment 24" in e["text"] for e in reader.evidence.values())
    assert reader.text_used <= config.query.text_tokens
    assert all(e["read_scope"] == "original_fragment" for e in reader.evidence.values())


def test_fragment_continuation_reads_new_text_without_restarting_node(local):
    config, store = local
    index = current_index(config, store, "structured")
    target = max(store.chunks(index["id"]), key=lambda c: len(c["text"]))
    reader = Reader(config, store, index)
    offset = 0
    for _ in range(50):
        result = reader.read(chunk_id=target["id"], offset=offset, max_tokens=25, include_images=False)
        first = result["results"][0]
        assert first["continuation_scope"] == "chunk"
        offset = first["next_offset"]
        if offset is None:
            break
    assert offset is None
    assert "".join(e["text"] for e in reader.evidence.values()) == target["text"]


def test_additive_enrichment_preserves_originals_and_never_cites_summary(configured):
    config, store, _ = configured
    base = current_index(config, store, "structured")
    originals = store.chunks(base["id"])
    assert not enrich(config, store)["failed"]
    enriched = chunk(config, store, "enriched")
    copies = store.chunks(enriched["id"])
    assert {c["text"] for c in originals} <= {c["text"] for c in copies if not c.get("enrichment_id")}
    summary = next(c for c in copies if c.get("enrichment_id") and c["kind"] == "table")
    reader = Reader(config, store, enriched)
    reader.question = "Test segment 24 2024"
    reader.read(chunk_id=summary["id"], include_images=False)
    assert reader.evidence
    assert all(e.get("chunk_id") != summary["id"] for e in reader.evidence.values())
    assert any("Test segment 24" in e["text"] for e in reader.evidence.values())
    node = store.get("nodes", summary["asset_ids"][0])
    key = enrichment_key(config, node)
    config.enrichment.retrieval_mode = "replace"
    assert enrichment_key(config, node) == key  # Retrieval policy does not regenerate paid descriptions.
    with pytest.raises(RagError, match="mode changed"):
        current_index(config, store, "enriched")


@pytest.mark.parametrize("recover", [True, False])
def test_json_has_exactly_five_retries_and_records_every_attempt(configured, recover):
    config, store, fake = configured
    count = 0

    def malformed(path, body):
        nonlocal count
        count += 1
        text = '{"ok": true}' if recover and count == 6 else '{"ok":'
        return 200, {"choices": [{"message": {"role": "assistant", "content": text}}]}

    fake.override = malformed
    api = ModelAPI(config, store, "json-recovery")

    def validate(m):
        return json.loads(m["content"])

    if recover:
        assert api.validated_chat("chat", [{"role": "user", "content": "Return JSON"}], validate)
    else:
        with pytest.raises(ValueError):
            api.validated_chat("chat", [{"role": "user", "content": "Return JSON"}], validate)
    assert count == 6
    assert [r["validation_attempt"] for r in store.calls("json-recovery")] == list(range(1, 7))


def test_nonrecoverable_http_and_budget_do_not_multiply_retries(configured):
    config, store, fake = configured
    fake.override = lambda p, b: (400, {"error": {"code": "invalid_request", "message": "SECRET"}})
    api = ModelAPI(config, store, "bad-request")
    with pytest.raises(APIError, match="HTTP 400"):
        api.validated_chat("chat", [{"role": "user", "content": "Return JSON"}], lambda m: None)
    assert len(fake.requests) == 1
    assert store.calls("bad-request")[0]["diagnostics"]["code"] == "invalid_request"
    assert "SECRET" not in json.dumps(store.calls("bad-request"))


def test_billing_error_blocks_later_roles_without_spending_requests(configured):
    config, store, fake = configured
    fake.override = lambda p, b: (402, {"error": {"code": "invalid_request_error"}})
    for role in ("chat", "judge", "vision"):
        with pytest.raises(APIError):
            ModelAPI(config, store, "billing:" + role).validated_chat(
                role, [{"role": "user", "content": "JSON"}], lambda m: None
            )
    assert len(fake.requests) == 1
    assert len(store.calls("billing:chat")) == 1
    assert store.calls("billing:judge") == []


def test_source_grouping_retains_matched_rows_and_a_uses_only_chunks(configured):
    config, store, _ = configured
    embed_index(config, store, "structured")
    index = current_index(config, store, "structured")
    search = Search(config, store, index, ModelAPI(config, store, "groups"))
    hits = search.search("Test segment 24 revenue 2024")
    assets = [tuple(c["asset_ids"]) for c in hits if c.get("asset_ids")]
    assert len(assets) == len(set(assets))
    assert len(hits) <= config.retrieval.final_k
    assert any(len(c["matched_chunk_ids"]) > 1 for c in hits)
    embed_index(config, store, "flat")
    index = current_index(config, store, "flat")
    result = QuestionRunner(config, store, index, ModelAPI(config, store, "direct-fragments")).run(
        {"query_id": "q", "query": "revenue"}, "A"
    )
    assert result["status"] == "ok"
    assert result["page_images_read"] == 0
    assert all(e["read_scope"] == "original_fragment" for e in result["evidence"])
    worker = QuestionRunner(config, store, index, ModelAPI(config, store, "source-locators"))
    _, payload = worker._search("revenue", previews=True)
    assert payload["page_index_origin"] == 0
    for hit in payload["hits"]:
        for source in hit["sources"]:
            doc = store.get("documents", payload["documents"][source["doc_id"]])
            assert 0 <= source["page_index"] < doc["pages"]
