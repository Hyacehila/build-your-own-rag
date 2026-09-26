"""CLI lifecycle; command options and adapters live in commands/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .commands import data, diagnostics, experiment
from .common import RagError
from .config import load_config
from .storage import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Financial document RAG: data preparation, four-arm experiments, and diagnostics.",
        epilog="Run '<command> --help' for options. Fact releases use 'python -m "
        "financial_rag.cleaning'; retrieval releases use 'python -m financial_rag.retrieval_release'.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML path (default: config.toml); relative work_dir is resolved against this file",
    )
    parser.set_defaults(read_only=False)
    commands = parser.add_subparsers(dest="command", required=True)
    data.register(commands)
    experiment.register(commands)
    diagnostics.register(commands)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        config = load_config(args.config)
        with Store(config.work_dir, read_only=args.read_only) as store:
            result = args.handler(args, config, store)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            if isinstance(result, dict) and (result.get("failed") or result.get("failures")):
                return 1
        return 0
    except ValidationError as e:
        errors = "; ".join(
            ".".join(map(str, r["loc"])) + ": " + r["msg"] for r in e.errors(include_input=False)
        )
        parser.exit(2, f"Invalid configuration: {errors}\n")
    except (RagError, FileNotFoundError) as e:
        parser.exit(2, f"Error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
