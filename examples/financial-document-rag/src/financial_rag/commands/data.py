"""Dataset preparation, parsing and base-index command adapters."""

from pathlib import Path


def register(commands):
    p = commands.add_parser(
        "download", help="Download the pinned benchmark and validate English page mapping"
    )
    p.set_defaults(handler=download)
    p = commands.add_parser("download-models", help="Download pinned local Docling layout/table weights")
    p.set_defaults(handler=download_models)
    p = commands.add_parser("inspect", help="Inspect native PDF text, layout and saved parse provenance")
    p.add_argument("--tables", action="store_true", help="Run native table detection on every page")
    p.set_defaults(handler=inspect)
    p = commands.add_parser("parse", help="Parse PDFs or reuse validated snapshots")
    p.add_argument("--parser", choices=["flat", "structured", "all"], default="all")
    p.add_argument("--documents", type=Path, help="Manifest of independently reviewed documents")
    p.set_defaults(handler=parse)
    p = commands.add_parser("chunk", help="Build flat, structured or enriched retrieval chunks")
    p.add_argument("--index", choices=["flat", "structured", "enriched", "all"], required=True)
    p.set_defaults(handler=chunk)


def download(args, config, store):
    from ..dataset import download

    return download(config, store)


def download_models(args, config, store):
    from financial_rag.document.models import download_models

    return download_models(config)


def inspect(args, config, store):
    from ..dataset import inspect

    return inspect(config, store, args.tables)


def parse(args, config, store):
    from financial_rag.document.parsing import parse

    if args.documents:
        if args.parser != "structured":
            from ..common import RagError

            raise RagError("--documents requires --parser structured")
        from financial_rag.document.imported import import_documents

        return import_documents(config, store, args.documents)
    kinds = ["flat", "structured"] if args.parser == "all" else [args.parser]
    results = [parse(config, store, kind) for kind in kinds]
    return {"results": results, "failed": [f for r in results for f in r["failed"]]}


def chunk(args, config, store):
    from financial_rag.indexing.chunking import chunk

    kinds = ["flat", "structured", "enriched"] if args.index == "all" else [args.index]
    return {"results": [chunk(config, store, kind) for kind in kinds]}
