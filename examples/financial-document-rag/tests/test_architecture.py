import copy
import json

import httpx
import numpy as np
import pytest

from financial_rag.api import APIError, ModelAPI, embedding_key
from financial_rag.budget import reserve, status
from financial_rag.chunking import chunk, current_index, make_chunk, merge_body_peers
from financial_rag.common import RagError, digest
from financial_rag.config import Config, Role, load_config
from financial_rag.facts import fact_content, migrate, normalize_document
from financial_rag.parsing import active_parses, load_docling, parse_signature, signature_matches
from financial_rag.release_security import secret_scan
from financial_rag.retrieval import embed_index
from financial_rag.sources import Reader
from financial_rag.tokenization import Tokenizer


def test_normalization_content_and_relations(local):
    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    original = load_docling(parsed)
    # Flatten an originally known tree while preserving existing heading levels.
    doc = original.model_copy(deep=True)
    from docling_core.types.doc import RefItem

    order = [
        item
        for item, _ in doc.iterate_items(with_groups=True, traverse_pictures=True)
        if item.self_ref != "#/body"
    ]
    doc.body.children = [RefItem(cref=i.self_ref) for i in order]
    for item in order:
        item.parent = RefItem(cref="#/body")
        item.children = []
    before = copy.deepcopy(doc.model_dump(mode="json"))
    normalized, _ = normalize_document(doc)
    assert fact_content(before) == fact_content(normalized.model_dump(mode="json"))
    assert doc.model_dump(mode="json") == before
    assert normalized.tables[0].parent.cref != "#/body"
    first = migrate(config, store)
    assert first["neural_parse_calls"] == 0 and first["foreign_key_errors"] == 0
    for kind in ("flat", "structured"):
        chunk(config, store, kind)
    ids = [current_index(config, store, k)["id"] for k in ("flat", "structured")]
    migrate(config, store)
    assert ids == [current_index(config, store, k)["id"] for k in ("flat", "structured")]
    active = active_parses(config, store, "structured")[0]
    assert active["raw_parse_id"] == parsed["id"]
    assert active["id"] != parsed["id"]
    assert not store.db.execute("PRAGMA foreign_key_check").fetchall()


def test_parse_identity_ignores_models_keys_and_directory(tmp_path, monkeypatch):
    config = Config(work_dir=tmp_path / "one")
    first = parse_signature(config, "structured")
    config.work_dir = tmp_path / "two"
    config.parsing.artifacts_path = str(tmp_path / "absent-weights")
    config.models.embedding = Role(model="different", base_url="https://example.org")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-private-token")
    assert signature_matches(first, parse_signature(config, "structured"))
    config.parsing.table_mode = "fast"
    assert not signature_matches(first, parse_signature(config, "structured"))


def test_merge_requires_body_parent_section_and_no_hidden_asset(local):
    config, store = local
    tokenizer = Tokenizer(config.tokenizer)
    pid = active_parses(config, store, "structured")[0]["id"]
    template = next(n for n in store.nodes(pid) if n["kind"] == "text")
    a = {**template, "id": "one", "reading_order": 1, "text": "First paragraph."}
    b = {**template, "id": "two", "reading_order": 3, "text": "Second paragraph."}

    def piece(n):
        return make_chunk(
            "\n".join([*n["headings"], n["text"]]), [n], "text", n["headings"], body_text=n["text"]
        )

    merged = merge_body_peers([piece(a), piece(b)], [a, b], config, tokenizer)
    assert len(merged) == 1 and merged[0]["node_ids"] == ["one", "two"]
    assert tokenizer.count(merged[0]["text"]) <= config.chunking.max_tokens
    hidden_asset = {**template, "id": "asset", "kind": "picture", "reading_order": 2}
    assert len(merge_body_peers([piece(a), piece(b)], [a, hidden_asset, b], config, tokenizer)) == 2
    for field in ("parent_id", "section_id"):
        other = {**b, field: "another"}
        assert len(merge_body_peers([piece(a), piece(other)], [a, other], config, tokenizer)) == 2


def test_structural_read_modes_budget_dedup_and_continuations(local):
    config, store = local
    index = current_index(config, store, "structured")
    nodes = store.nodes(index["parse_ids"][0])
    body = next(n for n in nodes if n["kind"] == "text")
    reader = Reader(config, store, index)
    result = reader.read(node_id=body["id"], mode="window", radius=2, include_images=False)
    for nid in result["expanded_nodes"]:
        n = store.get("nodes", nid)
        assert n["section_id"] == body["section_id"]
    used = reader.text_used
    reader.read(node_id=body["id"], mode="node", offset=2, include_images=False)
    assert reader.text_used == used
    assert "error" in reader.read(node_id=body["id"], mode="window", radius=9)
    assert "error" in reader.read(node_id=body["id"], mode="subtree", depth=5)
    parent = reader.read(node_id=body["id"], mode="parent", include_images=False)
    assert body["parent_id"] in parent["expanded_nodes"]
    root = next(n for n in nodes if n["ref"] == "#/body")
    tree = reader.read(node_id=root["id"], mode="subtree", depth=4, include_images=False)
    assert len(tree["expanded_nodes"]) <= 64
    assert not any(e["source_key"] == root["id"] for e in reader.evidence.values() if e["type"] == "text")
    small = config.model_copy(deep=True)
    small.query.text_tokens = 8
    limited = Reader(small, store, index)
    result = limited.read(node_id=body["id"], include_images=False)
    assert result["results"][0]["next_offset"] is not None
    assert limited.text_used <= 8


def test_sqlite_vec_ranking_matches_numpy_and_filters(configured):
    config, store, _ = configured
    embed_index(config, store, "flat")
    embed_index(config, store, "structured")
    index = current_index(config, store, "flat")
    chunks = store.chunks(index["id"])
    query = ModelAPI(config, store, "oracle").embed(["fictional revenue 2024"])[0]
    expected = []
    for c in chunks:
        v = store.vector(embedding_key(config.models.embedding, c["text"]))
        distance = 1 - float(
            np.dot(v.astype(float), query)
            / np.linalg.norm(v.astype(float))
            / np.linalg.norm(query.astype(float))
        )
        expected.append((distance, c["id"]))
    rows = store.dense(index["id"], digest(config.models.embedding.identity()), query, 30)
    assert [r["id"] for r in rows] == [cid for _, cid in sorted(expected)[:30]]
    assert store.dense(index["id"], "unmapped-model", query, 30) == []


def test_subtree_pagination_really_limits_64_items(local):
    config, store = local
    parsed = active_parses(config, store, "structured")[0]
    original = store.nodes(parsed["id"])
    template = next(n for n in original if n["kind"] == "text")
    root = copy.deepcopy(next(n for n in original if n["ref"] == "#/body"))
    pid = digest([parsed["id"], "wide-tree-test"])
    root.update(id=digest([pid, root["ref"]]), parse_id=pid, children=[])
    children = []
    for i in range(90):
        ref = f"#/texts/{i}"
        node = {
            **template,
            "id": digest([pid, ref]),
            "parse_id": pid,
            "ref": ref,
            "parent_ref": root["ref"],
            "children": [],
            "related_refs": [],
            "text": f"Fixture paragraph {i}.",
        }
        children.append(node)
        root["children"].append(ref)
    store.put_parse({**parsed, "id": pid}, [root, *children])
    reader = Reader(config, store, {"kind": "structured", "parse_ids": [pid]})
    first = reader.read(node_id=root["id"], mode="subtree", include_images=False)
    assert len(first["expanded_nodes"]) == 64 and first["next_node_start"] == 64
    second = reader.read(node_id=root["id"], mode="subtree", node_start=64, include_images=False)
    assert len(second["expanded_nodes"]) == 27 and second["next_node_start"] is None
    assert not set(first["expanded_nodes"]) & set(second["expanded_nodes"])


def test_exact_dimensions_and_thinking_tool_roundtrip(local):
    config, store = local
    config.models.embedding = Role(
        base_url="https://api.example.org", model="test", request_params={"dimensions": 1024}
    )
    text = "dimension validation"
    key = embedding_key(config.models.embedding, text)
    store.save_vectors([(key, np.ones(32, dtype=np.float32))])
    with pytest.raises(APIError, match="dimensions"):
        ModelAPI(config, store, "dims").embed([text])
    config.models.chat = Role(
        base_url="https://api.example.org",
        model="test",
        request_params={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    )
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Protocol-test reasoning field.",
                "tool_calls": [
                    {"id": "call1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                ],
            }
        else:
            assert body["messages"][1]["reasoning_content"] == "Protocol-test reasoning field."
            assert body["messages"][2]["role"] == "tool"
            message = {
                "role": "assistant",
                "content": '{"answer":"test","citations":[]}',
                "reasoning_content": "Done.",
            }
        return httpx.Response(
            200,
            json={
                "model": "actual-served-model",
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 2}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = ModelAPI(config, store, "thinking", client)
        tools = [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object", "properties": {}}},
            }
        ]
        messages = [{"role": "user", "content": "Call read."}]
        first = api.chat("chat", messages, tools)
        messages.extend([first, {"role": "tool", "tool_call_id": "call1", "content": "original evidence"}])
        api.chat("chat", messages, tools)
    assert store.calls("thinking")[-1]["response_model"] == "actual-served-model"
    assert store.calls("thinking")[-1]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2


def test_shared_ceiling_includes_retries_and_env_is_private(local, monkeypatch, tmp_path):
    config, store = local
    config.validation.ledger = tmp_path / "shared.sqlite"
    config.validation.deepseek_request_limit = 2
    config.models.chat = Role(base_url="https://api.example.org", model="test", retries=1)
    monkeypatch.setattr("financial_rag.api.time.sleep", lambda _: None)
    calls = []

    def error(request):
        calls.append(request)
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(error)) as client:
        with pytest.raises(APIError):
            ModelAPI(config, store, "retries", client).chat("chat", [{"role": "user", "content": "test"}])
        with pytest.raises(RagError, match="ceiling"):
            reserve(config.model_copy(deep=True), "other-workdir", "vision")
    assert len(calls) == status(config)["attempts_reserved"] == 2
    path = tmp_path / "config.toml"
    path.write_text('work_dir="data"\n', encoding="utf-8")
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=from-dotenv\nDEEPSEEK_API_KEY=unit-test-private-token\n"
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-wins")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    loaded = load_config(path)
    import os

    assert os.environ["DASHSCOPE_API_KEY"] == "environment-wins"
    assert os.environ["DEEPSEEK_API_KEY"] == "unit-test-private-token"
    assert "unit-test-private-token" not in canonical_config(loaded)
    store.set_meta("accidental-secret", "unit-test-private-token")
    with pytest.raises(RagError, match="Credential scan"):
        secret_scan(store.db)


def canonical_config(config):
    return json.dumps(config.portable_settings())
