import json
from types import SimpleNamespace

import pytest

from financial_rag.common import RagError
from financial_rag.config import Tokens
from financial_rag.query.agent_reading import AGENT_SYSTEM, TOOLS, FixedReading, parse_final
from financial_rag.query.structural_expansion import Structure
from financial_rag.tokenization import Tokenizer


def test_tolerant_final_normalizes_without_losing_unknown_citations():
    r = parse_final(
        '{"answer":"Fictional answer","citations":[{"evidence_id":"E1","note":"source"},"E1","E999"]}'
    )
    assert r["citations"] == ["E1", "E999"]
    assert "citation_object_normalized" in r["output_diagnostics"]
    assert parse_final(json.dumps(r))["output_diagnostics"] == r["output_diagnostics"]
    assert parse_final("A fictional answer [E2].")["citations"] == ["E2"]
    assert parse_final('{"answer":"An honest gap"}')["citations"] == []
    for v in ["", '{"answer":', '{"answer":12}']:
        with pytest.raises(ValueError):
            parse_final(v)


def test_parent_read_includes_child_sections_but_not_other_parent(monkeypatch):
    nodes = {}
    for i, (kind, parent, section) in enumerate(
        [
            ("section_header", None, "0"),
            ("section_header", "0", "1"),
            ("text", "1", "1"),
            ("section_header", "0", "3"),
            ("text", "3", "3"),
            ("section_header", None, "5"),
            ("text", "5", "5"),
        ]
    ):
        nodes[str(i)] = {
            "id": str(i),
            "parse_id": "p",
            "kind": kind,
            "parent_id": parent,
            "section_id": section,
            "reading_order": i,
            "text": "Fictional section " + str(i),
            "related_refs": [],
        }
    cat = SimpleNamespace(nodes=nodes, by_ref={}, chunks={"anchor": {"node_ids": ["2"]}})
    s = Structure(cat, Tokenizer(Tokens(kind="approx", name="fixture", estimated=True)))
    f = FixedReading(None, s)
    monkeypatch.setattr(f, "_read", lambda ids, **kw: {"ids": ids})
    result = f.read_parent("anchor")
    assert result["ids"] == ["0", "1", "2", "3", "4"]
    assert result["parent_section_ids"] == ["0"]
    with pytest.raises(RagError, match="No original parent"):
        f.read_parent("0")
    with pytest.raises(RagError, match="Unknown original"):
        f.read_parent("summary-entrance")


def test_fixed_read_has_no_model_selected_radius_and_pages_only_source_bound(monkeypatch):
    nodes = {
        str(i): {
            "id": str(i),
            "parse_id": "p",
            "kind": "text",
            "section_id": "s",
            "reading_order": i,
            "text": "fiction " * 3000,
            "related_refs": [],
            "sources": [{"doc_version": "v", "page_index": i // 2}],
        }
        for i in range(40)
    }
    s = Structure(
        SimpleNamespace(nodes=nodes, by_ref={}, chunks={"hit": {"node_ids": ["20"]}}),
        Tokenizer(Tokens(kind="approx", name="fixture", estimated=True)),
    )
    f = FixedReading(None, s)
    monkeypatch.setattr(f, "_read", lambda ids, **kw: {"ids": ids})
    assert set(f.read(["hit"])["ids"]) == {str(i) for i in range(12, 29)}
    monkeypatch.setattr(f, "_pages", lambda pages, **kw: pages)
    assert f.read_source_pages(["hit"]) == [("v", 10)]
    props = next(t["function"]["parameters"]["properties"] for t in TOOLS if t["function"]["name"] == "read")
    assert set(props) == {"source_ids", "include_images"}
    assert "read_parent" in AGENT_SYSTEM and "read_source_pages" in AGENT_SYSTEM


def test_initial_expansion_precedes_answer_and_pre_d_snapshot_isolated(configured):
    from financial_rag.common import digest
    from financial_rag.query.retrieval import embed_index
    from financial_rag.query.runner import run

    c, s, f = configured
    embed_index(c, s, "structured")
    run(c, s, "auto-fixed", ["D", "pre-D"], limit=1)
    d, p = sorted(s.records("auto-fixed"), key=lambda r: r["arm"])
    assert d["status"] == p["status"] == "ok"
    snap = s.cached(digest(["initial-evidence-v1", "auto-fixed", d["query_id"]]))
    assert d["initial_snapshot_sha256"] == p["initial_snapshot_sha256"] == snap["sha256"]
    assert p["evidence"] == [
        {k: v for k, v in e.items() if k != "data_url"} for e in snap["packet"]["evidence"].values()
    ]
    order = [t["tool"] for t in d["trace"]]
    auto = order.index("initial_fixed_expansion")
    assert d["trace"][auto]["actor"] == "controller"
    assert any(
        t["tool"] == "model_call" and t["phase"] == "answer_or_explore" for t in d["trace"][auto + 1 :]
    )


def test_d_object_citation_and_bad_coverage_do_not_cost_model_retry(configured):
    from financial_rag.query.retrieval import embed_index
    from financial_rag.query.runner import run

    c, s, f = configured
    embed_index(c, s, "structured")

    def override(path, body):
        if not body.get("tools"):
            return None
        return 200, {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "Fictional supplied original.",
                                "citations": [{"evidence_id": "E1"}],
                                "coverage": [],
                            }
                        ),
                    },
                }
            ]
        }

    f.override = override
    run(c, s, "tolerant", ["D"], limit=1)
    r = s.records("tolerant")[0]
    assert r["status"] == "ok" and r["model_calls"] == 2 and r["citations"] == ["E1"]
    assert r["coverage_diagnostic"]["issue"]
    assert len([b for path, b in f.requests if path.endswith("/chat/completions")]) == 2


def test_capacity_error_numbers_captured_without_provider_text(configured):
    from financial_rag.api import APIError, ModelAPI

    c, s, f = configured

    def reject(path, body):
        return 400, {
            "error": {
                "param": "input_tokens",
                "type": "BadRequestError",
                "message": "This model's maximum context length is 131072 tokens. However, you requested 16384 output tokens and your prompt contains at least 114689 input tokens. secret-provider-text",
            }
        }

    f.override = reject
    with pytest.raises(APIError) as error:
        ModelAPI(c, s, "cap").chat("chat", [{"role": "user", "content": "fiction"}], retry=False)
    assert error.value.diagnostics["context_window_tokens"] == 131072
    assert error.value.diagnostics["input_tokens_at_least"] == 114689
    assert "secret-provider-text" not in str(error.value)
    assert "secret-provider-text" not in json.dumps(s.calls("cap"))


def test_capacity_bound_includes_output_and_real_limit(configured):
    from financial_rag.query.context_capacity import Capacity

    c, s, f = configured
    c.query.context_window_tokens = 4096
    c.query.input_estimate_multiplier = 1.2
    cap = Capacity(c)
    messages = [
        {"role": "system", "content": "fiction"},
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "old fiction " * 5000},
        {"role": "user", "content": "latest source [E1]"},
    ]
    trace = []
    packed = cap.fit(messages, None, trace)
    assert cap.count(packed, None) <= 4096 and packed[-1] == messages[-1]
    assert trace[-1]["compacted_message_indices"] == [2]
    assert cap.original_batch_tokens(messages[:2], None) < 4096 - c.models.chat.request_params["max_tokens"]


def test_long_window_capacity_pagination_shared_by_tools_and_resumable(configured):
    from financial_rag.indexing.chunking import current_index
    from financial_rag.query.evidence_packet import OriginalCatalog
    from financial_rag.query.structural_expansion import StructuralReader

    c, s, _ = configured
    index = current_index(c, s, "structured")
    cat = OriginalCatalog(s, index)
    node = next(n for n in cat.nodes.values() if n["kind"] == "text" and n.get("sources"))
    node = {**node, "text": "Only fictional continuation content. " * 200}
    with s.db:
        s.db.execute("update nodes set payload=? where id=?", (json.dumps(node), node["id"]))
    cat.nodes[node["id"]] = node
    structure = Structure(cat, Tokenizer(c.tokenizer))
    reader = StructuralReader(c, s, index, unbounded=True)
    fixed = FixedReading(reader, structure, batch_tokens=100)
    first = fixed._read([node["id"]], include_images=False)
    assert first["continuation_id"] and reader.text_used <= 100 and reader.unbounded
    again = fixed._read([node["id"]], include_images=False)
    assert again["new_original_tokens_estimated"] == 0
    fixed.begin_turn()
    nxt = fixed.continue_read(first["continuation_id"])
    assert nxt["new_original_tokens_estimated"] > 0 and reader.text_used <= 200
    intervals = sorted((e["offset"], e["end_offset"]) for e in reader.evidence.values())
    assert intervals[0][1] == intervals[1][0]


def test_unknown_tool_feedback_does_not_repeat_physical_request(configured):
    from financial_rag.query.retrieval import embed_index
    from financial_rag.query.runner import run

    c, s, f = configured
    embed_index(c, s, "structured")
    issued = False

    def override(path, body):
        nonlocal issued
        if not body.get("tools") or issued:
            return None
        issued = True
        return 200, {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "bad-mode",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"mode":"section","source_ids":["fiction"]}',
                                },
                            }
                        ],
                    },
                }
            ]
        }

    f.override = override
    run(c, s, "tool-feedback", ["D"], limit=1)
    r = s.records("tool-feedback")[0]
    assert r["status"] == "ok" and r["model_calls"] == 3
    trace = next(t for t in r["trace"] if t.get("tool_call_id") == "bad-mode")
    assert "error" in trace["response"] and not trace["new_evidence"]
def test_zero_image_request_limit_omits_all_images(configured):
    from financial_rag.query.context_capacity import Capacity

    config, _, _ = configured
    config.query.agent_request_images = 0
    messages = [
        {"role": "system", "content": "Answer."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
    ]
    trace = []
    packed = Capacity(config).fit(messages, [], trace)
    assert packed[1]["content"][0]["type"] == "text"
    assert trace[-1]["image_pointers"] == 1
    assert messages[1]["content"][0]["type"] == "image_url"


def test_zero_total_image_limit_includes_agent_initial_read(configured):
    from test_five_arm_contract import prepare

    from financial_rag.query.runner import run

    config, store, _ = configured
    config.query.agent_total_images = 0
    config.query.agent_request_images = 0
    prepare(config, store)
    result = run(config, store, "agent-no-images", ["D", "pre-D"], limit=1)
    assert result["failures"] == 0
    rows = store.records("agent-no-images")
    assert all(not any(e["type"] == "image" for e in row["evidence"]) for row in rows)
