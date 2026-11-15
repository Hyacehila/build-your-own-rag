"""Pure source-text comparison helpers; no storage, retrieval or model dependencies."""

import re
import unicodedata
from collections import Counter


def lexical(text):
    """Diagnostic matching only: never save this normalized text back into facts."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u00ad", "")
    return Counter(re.findall(r"[a-z]+|\d+(?:[,.]\d+)*", text))


def numbers(text):
    text = unicodedata.normalize("NFKC", text)
    return Counter(v.replace(",", "") for v in re.findall(r"\d+(?:[,.]\d+)*", text))


def recovery(expected, observed):
    total = sum(expected.values())
    return sum((expected & observed).values()) / total if total else None
