from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any


class RagError(RuntimeError):
    """An actionable stage error, safe to display without an API response body."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation(*modules: str) -> str:
    """Fingerprint dependencies; costly parsing need not depend on report/API code."""
    here = Path(__file__).parent
    paths = [here / name for name in modules] if modules else list(here.rglob("*.py"))
    sources = {p.relative_to(here).as_posix(): sha256(p) for p in sorted(paths)}
    lock = here.parents[1] / "uv.lock"
    if lock.exists():
        sources["uv.lock"] = sha256(lock)
    return digest(sources)


def versions(*packages: str) -> dict:
    return {name: version(name) for name in packages}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)
