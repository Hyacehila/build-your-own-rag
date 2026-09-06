"""Run from this example directory with uv run demo.py. Original code: MIT."""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from fact_layer import FactError, Locator, build_bundle, json_bytes, load_bundle, read_node, structure_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and read a version-bound Docling document")
    parser.add_argument("action", choices=["build", "read", "validate"], nargs="?", default="build")
    parser.add_argument("--source", type=Path, default=Path("fixtures/product-review.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--document-id", default="product-review")
    parser.add_argument("--locator", type=Path, help="Saved locator JSON; defaults to OUTPUT/locator.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        if args.action == "build":
            build_bundle(args.source, args.output, args.document_id)
            _, document = load_bundle(args.output)
            print(structure_text(document), end="")
        if args.action == "validate":
            manifest, document = load_bundle(args.output)
            print(f"VALID: {manifest.document_id}, {len(document.texts)} texts, "
                  f"{len(document.tables)} tables, {len(document.pictures)} pictures")
        else:
            locator = Locator.model_validate_json(
                (args.locator or args.output / "locator.json").read_bytes()
            )
            print(json_bytes(read_node(args.output, locator)).decode("utf-8"), end="")
    except (FactError, OSError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
