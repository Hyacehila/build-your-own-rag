from __future__ import annotations

import hashlib
import json
import re

import httpx
import pytest
from support import prepare_test_source

from financial_rag.chunking import chunk
from financial_rag.config import Chunking, Config, Role, Tokens
from financial_rag.storage import Store


class FakeAPI:
    """In-process protocol double. Its answers/scores are intentionally fabricated."""

    def __init__(self):
        self.requests = []
        self.override = None
        self.usage = True
        self.url = None

    def response(self, path, body):
        self.requests.append((path, body))
        if self.override:
            result = self.override(path, body)
            if result is not None:
                return result
        usage = {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20} if self.usage else None
        if path.endswith("/embeddings"):
            data = []
            for i, text in enumerate(body["input"]):
                vector = [0.01] * 32
                for word in re.findall(r"\w+", text.lower()):
                    vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 32] += 1
                data.append({"index": i, "embedding": vector})
            # Returning reversed rows validates the adapter's use of response indices.
            return 200, {"data": data[::-1], "usage": usage}
        messages = body["messages"]
        text = json.dumps(messages)
        if isinstance(messages[0].get("content"), str) and messages[0]["content"].startswith(
            "Evaluate the candidate"
        ):
            content = json.dumps(
                {"correct": True, "reason": "Fabricated judge output for protocol test only"}
            )
        elif isinstance(messages[0].get("content"), str) and messages[0]["content"].startswith("Rewrite the user request"):
            content = json.dumps({"query": messages[1]["content"], "requested_items": [messages[1]["content"]]})
        elif "tools" in body and not re.search(r"\[E\d+\]", text):
            initial = next(
                m["content"]
                for m in messages
                if isinstance(m["content"], str) and m["content"].startswith("Initial search")
            )
            hit = json.loads(initial.split(": ", 1)[1])["hits"][0]
            return 200, {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": json.dumps({"chunk_id": hit["chunk_id"]}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": usage,
            }
        elif isinstance(messages[0].get("content"), str) and messages[0]["content"].startswith(
            "Answer financial"
        ):
            content = json.dumps(
                {
                    "answer": "Synthetic API answer for test verification only.",
                    "citations": list(dict.fromkeys(re.findall(r"\[(E\d+)\]", text)))[:2],
                }
            )
        else:
            content = "Fictional revenue table or chart. Business units, 2023 and 2024; USD million."
        if 'coverage must contain' in text or 'coverage has one entry' in text:
            value=json.loads(content)
            items=next(json.loads(p['text'].split('Requested items: ',1)[1])
                for m in messages if isinstance(m.get('content'),list)
                for p in m['content'] if p.get('type')=='text' and p.get('text','').startswith('Requested items: '))
            value['coverage']=[{'item_id':x['item_id'],'status':'supported' if value['citations'] else 'not_found',
                'evidence_ids':value['citations'],'scope_check':'Fictional test scope only',
                'gaps':'' if value['citations'] else 'No fictional evidence'} for x in items]
            content=json.dumps(value)
        return 200, {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }


@pytest.fixture
def fake(monkeypatch):
    fake = FakeAPI()
    fake.url = "http://test-provider.invalid/v1"
    original_client = httpx.Client

    def respond(request):
        if request.url.host != "test-provider.invalid" or request.method != "POST":
            raise AssertionError("Unexpected request outside the test provider")
        status, payload = fake.response(request.url.path, json.loads(request.content))
        return httpx.Response(status, json=payload, request=request)

    def client(*args, **kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = httpx.MockTransport(respond)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return fake


@pytest.fixture
def local(tmp_path, monkeypatch):
    config = Config(
        work_dir=tmp_path,
        tokenizer=Tokens(kind="approx", name="offline-test", estimated=True),
        chunking=Chunking(max_tokens=180, overlap=25),
    )
    with Store(tmp_path) as store:
        prepare_test_source(config, store, monkeypatch)
        chunk(config, store, "flat")
        chunk(config, store, "structured")
        yield config, store


@pytest.fixture
def configured(local, fake):
    config, store = local
    for name in ("embedding", "chat", "vision", "judge"):
        setattr(config.models, name, Role(base_url=fake.url, model="synthetic-" + name, retries=0))
    config.models.chat.request_params = {"max_tokens": 1024}
    proof = config.work_dir / "synthetic-capacity.json"
    proof.write_text(json.dumps({"model": config.models.chat.model, "base_url": fake.url,
        "verification": "operator_configuration", "source_reference": "In-process test provider; not a provider capability claim",
        "context_window_tokens": 1000000, "image_token_upper_bound": 4096,
        "message_overhead_tokens": 16, "request_overhead_tokens": 64, "text_encoding": "cl100k_base"}))
    config.query.capacity_evidence_path = proof
    return config, store, fake
