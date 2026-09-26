from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from financial_rag.chunking import chunk
from financial_rag.config import Chunking, Config, Role, Tokens
from financial_rag.fixture import create_fixture
from financial_rag.parsing import parse
from financial_rag.storage import Store


class FakeAPI:
    """Loopback-only protocol double. Its answers/scores are intentionally fabricated."""

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
        return 200, {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }


@contextmanager
def fake_server():
    fake = FakeAPI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, response = fake.response(self.path, body)
            raw = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def fake():
    with fake_server() as server:
        yield server


@pytest.fixture
def local(tmp_path):
    config = Config(
        work_dir=tmp_path,
        tokenizer=Tokens(kind="approx", name="offline-test", estimated=True),
        chunking=Chunking(max_tokens=180, overlap=25),
    )
    with Store(tmp_path) as store:
        create_fixture(config, store, structured_snapshot=True)
        assert not parse(config, store, "flat")["failed"]
        chunk(config, store, "flat")
        chunk(config, store, "structured")
        yield config, store


@pytest.fixture
def configured(local, fake):
    config, store = local
    for name in ("embedding", "chat", "vision", "judge"):
        setattr(config.models, name, Role(base_url=fake.url, model="synthetic-" + name, retries=0))
    return config, store, fake
