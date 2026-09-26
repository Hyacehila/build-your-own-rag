"""Lightweight v2 documentation and release-metadata check (no LFS payloads needed)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HANDOFF = {
    ".gitattributes",
    "finance-facts-v2.sqlite",
    "finance-retrieval-v2.sqlite",
    "finance-retrieval-v2.json",
    "finance-retrieval-v2-restore-check.json",
}
REQUIRED_DOCS = {
    "AGENTS.md",
    "README.md",
    "DESIGN.md",
    "OPERATIONS.md",
    "VALIDATION.md",
    "EXPERIMENT_HANDOFF.md",
    "DEVELOPMENT.md",
}
LINK = re.compile(r"!?\[[^\]]+\]\((<?[^)]+>?)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def slug(title: str) -> str:
    title = re.sub(r"<[^>]*>", "", title)
    title = re.sub(r"[^\w\-\s]", "", title.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", title.strip())


def headings(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    result = set()
    for title in HEADING.findall(path.read_text(encoding="utf-8")):
        base = slug(title)
        number = seen.get(base, 0)
        result.add(base if number == 0 else f"{base}-{number}")
        seen[base] = number + 1
    return result


def main() -> int:
    errors = []
    ignored = {".local", ".venv", ".pytest_cache", "node_modules", ".cache"}
    docs = sorted(
        path for path in ROOT.rglob("*.md") if not ignored.intersection(path.relative_to(ROOT).parts)
    )
    for name in sorted(REQUIRED_DOCS):
        if not (ROOT / name).is_file():
            errors.append(f"missing required project document: {name}")
    for doc in docs:
        source = doc.read_text(encoding="utf-8")
        source = re.sub(r"^```.*?^```", "", source, flags=re.MULTILINE | re.DOTALL)
        for raw in LINK.findall(source):
            target = raw.strip("<>").split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("//"):
                continue
            relative = unquote(parsed.path)
            if any(
                part in {".local", ".venv", "outputs", "runs", "model_weights"}
                for part in Path(relative).parts
            ):
                errors.append(f"{doc.relative_to(ROOT)}: ignored-directory link: {target}")
                continue
            destination = (doc.parent / relative).resolve() if relative else doc
            if not destination.is_file():
                errors.append(f"{doc.relative_to(ROOT)}: missing link target: {target}")
            elif parsed.fragment and unquote(parsed.fragment) not in headings(destination):
                errors.append(f"{doc.relative_to(ROOT)}: missing heading: {target}")

    actual = {path.name for path in (ROOT / "handoff").iterdir()}
    if actual != EXPECTED_HANDOFF:
        errors.append(f"unexpected handoff inventory: {sorted(actual ^ EXPECTED_HANDOFF)}")
    report = json.loads((ROOT / "handoff/finance-retrieval-v2.json").read_text(encoding="utf-8"))
    restored = json.loads(
        (ROOT / "handoff/finance-retrieval-v2-restore-check.json").read_text(encoding="utf-8")
    )
    release_id = report["report"]["fact_release_id"]
    indices = {item["kind"]: item["chunks"] for item in report["report"]["indices"]}
    validation = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
    expected_rows = {
        "事实发布 ID": f"`{release_id}`",
        "A 平面切片": f"{indices['flat']:,}",
        "B 结构切片": f"{indices['structured']:,}",
    }
    for label, value in expected_rows.items():
        if f"| {label} | {value} |" not in validation:
            errors.append(f"VALIDATION.md: {label} differs from v2 handoff JSON")
    if restored["prepared"]["fact_release_id"] != release_id:
        errors.append("v2 restore and export fact release IDs differ")
    if restored["check"]["status"] != "passed" or report["report"]["status"] != "passed":
        errors.append("v2 retrieval checks did not pass")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Documentation OK: {len(docs)} Markdown files; fact {release_id[:12]}; A/B {indices['flat']}/{indices['structured']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
