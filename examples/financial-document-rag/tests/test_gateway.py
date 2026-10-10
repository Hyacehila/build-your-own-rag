import json
import re

import pytest

from financial_rag.common import RagError
from financial_rag.release_security import secret_scan
from financial_rag.validation import protocol_check


@pytest.mark.parametrize("reasoning", [False, True])
def test_gateway_protocol_accepts_optional_reasoning_and_replays_it(configured, reasoning):
    config, store, fake = configured
    config.validation.ledger = config.work_dir / "probe-budget.sqlite"

    def reply(path, body):
        if not path.endswith("/chat/completions"):
            return None
        messages = body["messages"]
        if any(m["role"] == "tool" for m in messages):
            previous = next(m for m in messages if m["role"] == "assistant")
            assert (previous.get("reasoning_content") == "Probe reasoning") is reasoning
            message = {
                "role": "assistant",
                "content": json.dumps({"answer": "USD 150 million", "citations": ["E1"]}),
            }
        else:
            prompt = messages[1]["content"][0]["text"]
            version = re.search(r"doc_version=([a-f0-9]+)", prompt).group(1)
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-original",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps(
                                {"doc_version": version, "page_index": 0, "include_images": False}
                            ),
                        },
                    }
                ],
            }
            if reasoning:
                message["reasoning_content"] = "Probe reasoning"
        return 200, {"choices": [{"message": message}]}

    fake.override = reply
    result = protocol_check(config, store)
    assert result["status"] == "passed"
    assert result["tool_roundtrip"] is True
    assert result["reasoning_content_returned"] is reasoning
    assert result["reasoning_roundtrip"] == (True if reasoning else None)


def test_release_scan_covers_new_gateway_token(local, monkeypatch):
    _, store = local
    monkeypatch.setenv("NEW_API_KEY", "test-gateway-private-token")
    store.set_meta("accidental-gateway-token", "test-gateway-private-token")
    with pytest.raises(RagError, match="Credential scan") as caught:
        secret_scan(store.db)
    assert "test-gateway-private-token" not in str(caught.value)
