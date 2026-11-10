import json

import pytest

from financial_rag.api import ModelAPI
from financial_rag.common import RagError
from financial_rag.release_security import secret_scan


@pytest.mark.parametrize("reasoning", [False, True])
def test_gateway_protocol_accepts_optional_reasoning_and_replays_it(configured, reasoning):
    config, store, fake = configured

    def reply(path, body):
        if not path.endswith("/chat/completions"):
            return None
        messages = body["messages"]
        if any(m["role"] == "tool" for m in messages):
            previous = next(m for m in messages if m["role"] == "assistant")
            assert (previous.get("reasoning_content") == "Probe reasoning") is reasoning
            message = {
                "role": "assistant",
                "content": json.dumps({"answer": "value", "citations": ["E1"]}),
            }
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-original",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"source_id": "source-1"}),
                        },
                    }
                ],
            }
            if reasoning:
                message["reasoning_content"] = "Probe reasoning"
        return 200, {"choices": [{"message": message}]}

    fake.override = reply
    api = ModelAPI(config, store, "gateway-test")
    messages = [{"role": "system", "content": "Read before answering"},
                {"role": "user", "content": "Find the value"}]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    first = api.chat("chat", messages, tools=tools)
    assert bool(first.get("reasoning_content")) is reasoning
    messages.extend([first, {"role": "tool", "tool_call_id": "read-original", "content": "[E1] value"}])
    final = api.chat("chat", messages)
    assert json.loads(final["content"])["citations"] == ["E1"]


def test_release_scan_covers_new_gateway_token(local, monkeypatch):
    _, store = local
    monkeypatch.setenv("NEW_API_KEY", "test-gateway-private-token")
    store.set_meta("accidental-gateway-token", "test-gateway-private-token")
    with pytest.raises(RagError, match="Credential scan") as caught:
        secret_scan(store.db)
    assert "test-gateway-private-token" not in str(caught.value)
