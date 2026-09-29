from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx
import numpy as np

from .common import RagError, digest
from .config import Config, Role
from .storage import Store


class APIError(RagError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def embedding_key(role: Role, text: str) -> str:
    return digest({"type": "embedding-v1", "role": role.identity(), "input": text})


def token_usage(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    result = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
    ):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[name] = value
    for name in ("prompt_tokens_details", "completion_tokens_details"):
        detail = raw.get(name)
        if isinstance(detail, dict):
            result[name] = {
                k: v
                for k, v in detail.items()
                if k in {"cached_tokens", "reasoning_tokens", "audio_tokens", "image_tokens", "text_tokens"}
                and isinstance(v, int)
                and not isinstance(v, bool)
                and v >= 0
            }
    return result or None


class ModelAPI:
    def __init__(self, config: Config, store: Store, context: str, client: httpx.Client | None = None):
        self.config, self.store, self.context = config, store, context
        self.client = client

    def _post(self, role_name: str, endpoint: str, payload: dict, retry: bool = True) -> dict:
        role = getattr(self.config.models, role_name)
        role.require(role_name)
        account = (role.base_url.rstrip("/"), role.api_key_env)
        if account in getattr(self.store, "billing_blocks", set()):
            raise APIError(
                "Provider billing blocked after HTTP 402; no new request sent. Resume in a new process after resolving billing.",
                status_code=402,
            )
        headers = {"Content-Type": "application/json"}
        if role.api_key_env:
            headers["Authorization"] = "Bearer " + os.environ[role.api_key_env]
        body = {**role.request_params, "model": role.model, **payload}
        for attempt in range((role.retries if retry else 0) + 1):
            from .budget import reserve

            reserve(self.config, self.context, role_name)
            start, status, usage = time.perf_counter(), None, None
            error = response_model = None
            diagnostics = {}
            try:
                if self.client is not None:
                    response = self.client.post(
                        role.base_url.rstrip("/") + endpoint,
                        json=body,
                        headers=headers,
                        timeout=role.timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=role.timeout_seconds) as client:
                        response = client.post(
                            role.base_url.rstrip("/") + endpoint, json=body, headers=headers
                        )
                status = response.status_code
                if response.is_success:
                    try:
                        data = response.json()
                    except ValueError as e:
                        raise APIError("API returned invalid JSON") from e
                    if not isinstance(data, dict):
                        raise APIError("API returned a non-object JSON response")
                    usage = token_usage(data.get("usage"))
                    response_model = data.get("model") if isinstance(data.get("model"), str) else None
                    return data
                error = f"HTTP {status}"
                if status == 402:
                    # Shared across chat/vision/judge in this Store session, never persisted
                    # across a user-authorized restart after replenishing the account.
                    self.store.billing_blocks = getattr(self.store, "billing_blocks", set()) | {account}
                try:
                    detail = response.json().get("error", {})
                    if isinstance(detail, dict):
                        diagnostics = {
                            k: v
                            for k, v in detail.items()
                            if k in {"code", "type", "param"}
                            and isinstance(v, str)
                            and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", v)
                        }
                except (ValueError, AttributeError):
                    pass
                request_id = response.headers.get("x-request-id", "")
                if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
                    diagnostics["request_id"] = request_id
                if status not in (408, 429, 500, 502, 503, 504) or attempt >= (role.retries if retry else 0):
                    raise APIError(
                        f"{role_name}: {error}. Check endpoint/model/capabilities; response body omitted.",
                        status_code=status,
                    )
            except httpx.RequestError as e:
                error = type(e).__name__
                if attempt >= (role.retries if retry else 0):
                    raise APIError(f"{role_name}: {error}; request failed.") from e
            except APIError as e:
                error = str(e)
                raise
            finally:
                # No headers, keys, request bodies or raw API errors in the usage ledger.
                self.store.log_call(
                    self.context,
                    role_name,
                    {
                        "model": role.model,
                        "response_model": response_model,
                        "identity": digest(role.identity()),
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                        "status_code": status,
                        "error": error,
                        "seconds": time.perf_counter() - start,
                        "usage": usage,
                        "diagnostics": diagnostics,
                        "request_shape": {
                            "messages": len(payload.get("messages", [])),
                            "roles": [m.get("role") for m in payload.get("messages", [])],
                            "tools": bool(payload.get("tools")),
                            "max_tokens": body.get("max_tokens"),
                        },
                    },
                )
            time.sleep(min(2**attempt, 8))
        raise APIError("Request failed")

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        role = self.config.models.embedding
        if not role.configured():
            role.require("embedding")
        unique = {embedding_key(role, t): t for t in texts}
        vectors = {key: self.store.vector(key) for key in unique}
        missing = [key for key in unique if vectors[key] is None]
        if missing:
            role.require("embedding")
        known_dims = {len(v) for v in vectors.values() if v is not None}
        if len(known_dims) > 1:
            raise APIError("Embedding cache contains inconsistent dimensions; change cache_revision.")
        expected = role.request_params.get("dimensions")
        if expected is not None and any(d != expected for d in known_dims):
            raise APIError("Cached embedding dimensions differ from the configured dimensions.")
        dimensions = expected if expected is not None else next(iter(known_dims), None)
        batch_size = (
            min(role.batch_size, 20) if "dashscope.aliyuncs.com" in role.base_url else role.batch_size
        )
        for offset in range(0, len(missing), batch_size):
            keys = missing[offset : offset + batch_size]
            result = self._post(
                "embedding", "/embeddings", {"input": [unique[k] for k in keys], "encoding_format": "float"}
            )
            try:
                rows = result["data"]
                if len(rows) != len(keys) or sorted(r["index"] for r in rows) != list(range(len(keys))):
                    raise ValueError("missing, duplicated or invalid indices")
                batch = []
                for row in sorted(rows, key=lambda r: r["index"]):
                    if not isinstance(row["embedding"], list):
                        raise ValueError("embedding must be a float array")
                    v = np.asarray(row["embedding"], dtype=np.float32)
                    if v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
                        raise ValueError("embedding must be nonzero and finite")
                    dimensions = len(v) if dimensions is None else dimensions
                    if len(v) != dimensions:
                        raise ValueError("inconsistent dimensions; change cache_revision if model changed")
                    batch.append((keys[row["index"]], v))
            except (KeyError, TypeError, ValueError) as e:
                known = {
                    "missing, duplicated or invalid indices",
                    "embedding must be a float array",
                    "embedding must be nonzero and finite",
                    "inconsistent dimensions; change cache_revision if model changed",
                }
                reason = str(e) if str(e) in known else f"{type(e).__name__} (malformed vector)"
                safe_error = f"Invalid embedding response: {reason}; batch not cached."
                self.store.call_validation_error(self.context, "embedding", safe_error)
                raise APIError(safe_error) from e
            self.store.save_vectors(batch)
            vectors.update(batch)
            if len(missing) > 100 and (offset // batch_size % 10 == 0 or offset + batch_size >= len(missing)):
                print(
                    f"Embedding cached: {min(offset + batch_size, len(missing))}/{len(missing)} new inputs",
                    flush=True,
                )
        return [vectors[embedding_key(role, t)] for t in texts]

    def chat(
        self,
        role_name: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        retry: bool = True,
        tool_choice=None,
        json_output=False,
    ) -> dict:
        payload: dict = {"messages": messages, "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice or "auto")
        elif json_output:
            payload["response_format"] = {"type": "json_object"}
        result = self._post(role_name, "/chat/completions", payload, retry=retry)
        try:
            choices = result["choices"]
            if not choices:
                raise ValueError("no choices")
            message = choices[0]["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("expected assistant message")
            self.store.annotate_call(
                self.context,
                role_name,
                {
                    "finish_reason": choices[0].get("finish_reason"),
                    "response_message": {
                        k: message[k]
                        for k in ("role", "content", "tool_calls", "reasoning_content")
                        if k in message
                    },
                },
            )
            if choices[0].get("finish_reason") in {"length", "content_filter"}:
                raise ValueError(f"incomplete completion: {choices[0]['finish_reason']}")
            if not message.get("tool_calls") and not isinstance(message.get("content"), str):
                raise ValueError("missing assistant content")
            if message.get("tool_calls") and not tools:
                raise ValueError("tool calls returned in a tools-disabled final answer")
            return {
                k: message[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in message
            }
        except (KeyError, TypeError, ValueError) as e:
            self.store.call_validation_error(self.context, role_name, "Invalid chat response: " + str(e))
            raise APIError(f"Invalid chat response: {e}") from e

    def validated_chat(self, role_name, messages, validate, tools=None, *, json_output=None):
        """First request + at most five retries, without nested HTTP retries.

        Validate before executing tools so retries cannot repeat tool side effects.
        Logical agent turns and physical requests are counted separately.
        """
        role = getattr(self.config.models, role_name)
        current = list(messages)
        for attempt in range(max(role.json_retries, role.retries) + 1):
            try:
                message = self.chat(
                    role_name,
                    current,
                    tools,
                    retry=False,
                    json_output=role.json_mode if json_output is None else json_output,
                )
                validate(message)
                self.store.annotate_call(self.context, role_name, {"validation_attempt": attempt + 1})
                return message
            except (APIError, ValueError, TypeError, KeyError) as error:
                self.store.annotate_call(
                    self.context,
                    role_name,
                    {
                        "validation_attempt": attempt + 1,
                        "validation_error": type(error).__name__,
                    },
                )
                code = getattr(error, "status_code", None)
                if code is not None and code not in {408, 429, 500, 502, 503, 504}:
                    raise
                limit = (
                    role.retries if isinstance(error, APIError) and code is not None else role.json_retries
                )
                if attempt >= limit:
                    raise
                if isinstance(error, APIError):
                    time.sleep(min(2**attempt, 8))
                else:
                    # Preserve evidence and reasoning history; never replay malformed tools.
                    instruction = (
                        "Your previous response was empty or invalid. Return a complete, nonempty "
                        "retrieval description using only the supplied source; do not invent facts."
                        if json_output is False
                        else "Your previous response failed format/citation validation. Return valid JSON "
                        "matching the required schema, or valid tool arguments. Use only supplied "
                        "evidence IDs; do not invent facts. No markdown fences."
                    )
                    current = [
                        *messages,
                        {
                            "role": "user",
                            "content": instruction,
                        },
                    ]


def usage_summary(calls: list[dict]) -> dict:
    roles = {}
    for role in sorted({c["role"] for c in calls}):
        values = [c for c in calls if c["role"] == role]
        roles[role] = {
            "requests": len(values),
            "failed_requests": sum(bool(c["error"]) for c in values),
            "usage_missing_requests": sum(c["usage"] is None for c in values),
        }
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            observed = [c["usage"][name] for c in values if c["usage"] is not None and name in c["usage"]]
            roles[role][name + "_reported"] = sum(observed) if observed else None
    return roles
