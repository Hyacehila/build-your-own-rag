"""Build the original financial teaching fixture without models or historical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_rag.chunking import chunk
from financial_rag.common import RagError, write_json
from financial_rag.config import load_config
from financial_rag.fixture import create_fixture
from financial_rag.parsing import active_parses, load_docling, parse
from financial_rag.sources import Reader
from financial_rag.storage import Store

PROJECT = Path(__file__).resolve().parents[1]


def build(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise RagError(f"Output already exists; choose another --output directory: {output}")
    config = load_config(PROJECT / "config.offline.toml")
    config.work_dir = output
    with Store(output) as store:
        fixture = create_fixture(config, store, structured_snapshot=True)
        flat = parse(config, store, "flat")
        if flat["failed"]:
            raise RagError("The fictional PDF could not be read as native text.")
        indices = {kind: chunk(config, store, kind) for kind in ("flat", "structured")}
        parsed = active_parses(config, store, "structured")[0]
        document = load_docling(parsed)
        write_json(output / "document.json", document.model_dump(mode="json", by_alias=True))
        tree = [f"{'  ' * depth}{item.self_ref} [{item.label.value}] "
                f"{getattr(item, 'text', '') or getattr(item, 'name', '') or ''}".rstrip()
                for item, depth in document.iterate_items(with_groups=True)]
        (output / "structure.txt").write_text("\n".join(tree) + "\n", encoding="utf-8")
        previews = []
        for kind, index in indices.items():
            nodes = {node["id"]: node for pid in index["parse_ids"] for node in store.nodes(pid)}
            for piece in store.chunks(index["id"]):
                previews.append({"index": kind, "kind": piece["kind"],
                                 "headings": piece["headings"], "text": piece["text"],
                                 "refs": [nodes[nid]["ref"] for nid in piece["node_ids"]],
                                 "sources": piece["sources"]})
        write_json(output / "chunks.json", previews)
        nodes = store.nodes(parsed["id"])
        table = next(node for node in nodes if node["kind"] == "table")
        reader = Reader(config, store, indices["structured"])
        read = reader.read(node_id=table["id"], include_images=False)
        if "error" in read:
            raise RagError(read["error"])
        write_json(output / "readback.json", {"ref": table["ref"], "read": read,
                                               "evidence": reader.public_evidence()})
    return {"output": str(output), "pdf": fixture["pdf"], "synthetic": True,
            "chunk_counts": {kind: index["chunk_count"] for kind, index in indices.items()},
            "note": "Known-layout teaching fixture; does not test neural PDF parsing or answer quality."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / ".local/teaching")
    args = parser.parse_args()
    try:
        result = build(args.output)
    except RagError as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
