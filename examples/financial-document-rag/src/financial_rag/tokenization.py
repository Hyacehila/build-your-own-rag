from __future__ import annotations

import re
from typing import Any

from .config import Tokens


class ApproxEncoding:
    """Offline, explicitly approximate counter; never passed off as provider usage."""

    def encode(self, text: str, **_: Any) -> list[str]:
        return re.findall(r"\s+|[A-Za-z0-9_]{1,4}|[^\w\s]|\w", text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


class Tokenizer:
    def __init__(self, settings: Tokens):
        self.settings = settings
        if settings.kind == "tiktoken":
            import tiktoken

            self.encoding = tiktoken.get_encoding(settings.name)
        elif settings.kind == "huggingface":
            from transformers import AutoTokenizer

            self.encoding = AutoTokenizer.from_pretrained(
                settings.name, revision=settings.revision, trust_remote_code=False
            )
        else:
            self.encoding = ApproxEncoding()

    def count(self, text: str) -> int:
        if self.settings.kind == "huggingface":
            return len(self.encoding.encode(text, add_special_tokens=False))
        if self.settings.kind == "tiktoken":
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(self.encoding.encode(text))

    def prefix(self, text: str, limit: int) -> str:
        """Cut on character boundaries, preserving valid Unicode and the hard budget."""
        if limit <= 0:
            return ""
        if self.count(text) <= limit:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    def docling(self, maximum: int):
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
        from pydantic import ConfigDict

        class Adapter(BaseTokenizer):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            counter: Any
            maximum: int

            def count_tokens(self, text: str) -> int:
                return self.counter.count(text)

            def get_max_tokens(self) -> int:
                return self.maximum

            def get_tokenizer(self):
                # semchunk accepts a token-counting function as well as a tokenizer.
                return self.counter.count

        return Adapter(counter=self, maximum=maximum)


def recursive_spans(text: str, tokenizer: Tokenizer, maximum: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Recursive separators, then merge with a token-limited overlapping suffix.

    Offsets always point into the untouched concatenated source, including across pages.
    """
    if not 0 <= overlap < maximum:
        raise ValueError("Require 0 <= overlap < maximum")

    def pieces(start: int, end: int, separators: list[str]) -> list[tuple[int, int]]:
        value = text[start:end]
        if tokenizer.count(value) <= maximum:
            return [(start, end)] if value else []
        if not separators:
            result = []
            while start < end:
                part = tokenizer.prefix(text[start:end], maximum)
                if not part:
                    raise ValueError("Token budget cannot hold a single character")
                result.append((start, start + len(part)))
                start += len(part)
            return result
        sep, *rest = separators
        bounds = [start] + [start + m.end() for m in re.finditer(re.escape(sep), value)]
        if bounds[-1] != end:
            bounds.append(end)
        return [p for a, b in zip(bounds, bounds[1:]) for p in pieces(a, b, rest)]

    units = pieces(0, len(text), ["\n\n", "\n", ". ", " "])
    result: list[tuple[int, int]] = []
    for a, b in units:
        if not result:
            result.append((a, b))
        elif tokenizer.count(text[result[-1][0] : b]) <= maximum:
            result[-1] = (result[-1][0], b)
        else:
            prev_start, prev_end = result[-1]
            low, high = prev_start, prev_end
            while low < high:
                mid = (low + high) // 2
                if tokenizer.count(text[mid:prev_end]) <= overlap and tokenizer.count(text[mid:b]) <= maximum:
                    high = mid
                else:
                    low = mid + 1
            result.append((low, b))
    return result
