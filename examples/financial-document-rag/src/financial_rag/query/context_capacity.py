"""Estimate request size against the configured model context window."""

import copy
from functools import lru_cache

from financial_rag.common import RagError, canonical
from financial_rag.tokenization import Tokenizer


class Capacity:
    def __init__(self, config):
        self.limit = config.query.context_window_tokens
        if self.limit is None:
            raise RagError("Set query.context_window_tokens for the actual chat model before running D")
        self.output = config.models.chat.request_params.get("max_tokens")
        if not isinstance(self.output, int) or self.output <= 0 or self.output >= self.limit:
            raise RagError("Declare a valid provider output reserve")
        self.tokenizer = Tokenizer(config.tokenizer)
        self.multiplier = config.query.input_estimate_multiplier
        self.image_limit = config.query.agent_request_images
        self.identity = {
            "model": config.models.chat.model,
            "context_window_tokens": self.limit,
            "output_reserve_tokens": self.output,
            "input_estimate_multiplier": self.multiplier,
            "tokenizer": config.tokenizer.model_dump(),
            "image_counting": "serialized_payload_tokens_proxy",
            "counting_is_estimated": True,
        }

    @lru_cache(maxsize=32)
    def text_count(self, text):
        return self.tokenizer.count(text)

    def count(self, messages, tools):
        # Image payload length is a proxy, not provider billing.
        inline = 0

        def count_payloads(x):
            nonlocal inline
            if isinstance(x, dict):
                if x.get("type") == "image_url":
                    inline += self.text_count(canonical(x))
                    return {"type": "image_url", "counted_separately": True}
                return {k: count_payloads(v) for k, v in x.items()}
            return [count_payloads(v) for v in x] if isinstance(x, list) else x

        text = canonical({"messages": count_payloads(messages), "tools": tools or []})
        return int((self.text_count(text) + inline + len(messages) * 16 + 64) * self.multiplier) + self.output

    def original_batch_tokens(self, base_messages, tools):
        # Capacity-derived page size, not a cumulative original-reading budget.
        # Half of the available space accommodates source identities, tool results,
        # existing context and tokenizer mismatch; the rest is rereadable via cursor.
        return max(1, int((self.limit - self.count(base_messages, tools)) / (2 * self.multiplier)))

    def after_rejection(self, messages, tools, trace, diagnostics):
        maximum = diagnostics.get("context_window_tokens")
        if isinstance(maximum, int) and maximum > self.output:
            self.limit = min(self.limit, maximum)
        trace.append(
            {
                "tool": "capacity_rejection",
                "diagnostics": diagnostics,
                "action": "one_compacted_physical_retry",
                "context_window_tokens": self.limit,
            }
        )
        return self.fit(messages, tools, trace, force=True)

    def fit(self, messages, tools, trace, force=False):
        packed = copy.deepcopy(messages)
        # Earlier images remain available through the evidence store.
        image_parts = [
            (i, j)
            for i, m in enumerate(packed)
            if isinstance(m.get("content"), list)
            for j, part in enumerate(m["content"])
            if part.get("type") == "image_url"
        ]
        image_pointers = 0
        earlier = image_parts[: -self.image_limit] if self.image_limit else image_parts
        recent = image_parts[-self.image_limit :] if self.image_limit else []
        for i, j in earlier:
            image_pointers += 1
            packed[i]["content"][j] = {
                "type": "text",
                "text": "[Earlier original image retained in evidence store; read its source again to display.]",
            }
        # For an explicitly user-configured capacity with unknown image token rules,
        # count serialized image payloads as a transparent proxy, not provider billing.
        # Keep newest images that fit; full originals remain redisplayable in the vault.
        for i, j in recent:
            if not force and self.count(packed, tools) <= self.limit:
                break
            image_pointers += 1
            packed[i]["content"][j] = {
                "type": "text",
                "text": "[Original image retained in evidence store; omitted from this request under the declared capacity proxy.]",
            }
        compacted = []
        # Never compact system, original question, or latest instruction/tool result.
        for i in range(2, max(2, len(packed) - 1)):
            if not force and self.count(packed, tools) <= self.limit:
                break
            m = packed[i]
            if m["role"] in {"tool", "user"} and m.get("content"):
                import re

                text = canonical(m["content"])
                ids = sorted(set(re.findall(r"\bE\d+\b", text)))
                m["content"] = (
                    "Earlier tool content remains in the evidence store. Source IDs: "
                    + ", ".join(ids)
                    + ". Use read_evidence to reread originals; this pointer is not evidence."
                )
                compacted.append(i)
        count = self.count(packed, tools)
        trace.append(
            {
                "tool": "context_capacity",
                "count_including_output_reserve": count,
                "context_window_tokens": self.limit,
                "compacted_message_indices": compacted,
                "image_pointers": image_pointers,
                "capacity_source": "configuration",
                "counting_is_estimated": True,
            }
        )
        if count > self.limit:
            raise RagError(
                "Request cannot fit the declared gateway capacity without losing its current input"
            )
        return packed
