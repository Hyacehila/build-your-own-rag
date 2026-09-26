"""Enhancement, embedding, answering and evaluation command adapters."""

import json

from ..common import write_json
from ..config import ARMS


def register(commands):
    p = commands.add_parser(
        "enrich", help="Generate picture descriptions and whole-table retrieval summaries"
    )
    p.add_argument("--node-id", action="append", help="Restrict enhancement to selected active assets")
    p.set_defaults(handler=enrich)
    p = commands.add_parser("estimate-embedding", help="Estimate distinct uncached embedding input costs")
    p.set_defaults(handler=estimate)
    p = commands.add_parser("embed", help="Embed chunks, reusing exact cached inputs")
    p.add_argument("--index", choices=["flat", "structured", "enriched", "all"], default="all")
    p.set_defaults(handler=embed)
    p = commands.add_parser("run", help="Run or resume an immutable A/B/C/D experiment")
    p.add_argument("--run", required=True, help="Immutable run ID; reuse only to resume the same settings")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-errors", action="store_true")
    p.set_defaults(handler=run)
    p = commands.add_parser("evaluate", help="Score a run against its frozen labels")
    p.add_argument("--run", required=True)
    p.set_defaults(handler=evaluate)
    p = commands.add_parser("report", help="Export an evaluated run's report")
    p.add_argument("--run", required=True)
    p.set_defaults(handler=report)
    p = commands.add_parser("usage", help="Export recorded API calls and tokens")
    p.set_defaults(handler=usage)


def enrich(args, config, store):
    from ..enrichment import enrich

    result = enrich(config, store, args.node_id)
    write_json(config.work_dir / "enrichment.json", result)
    return result


def estimate(args, config, store):
    from ..retrieval import estimate

    result = estimate(config, store)
    write_json(config.work_dir / "embedding-estimate.json", result)
    return result


def embed(args, config, store):
    from ..retrieval import embed_index

    kinds = ["flat", "structured", "enriched"] if args.index == "all" else [args.index]
    return {"results": [embed_index(config, store, kind) for kind in kinds]}


def run(args, config, store):
    from ..runner import run

    return run(config, store, args.run, list(dict.fromkeys(args.arms)), args.limit, args.retry_errors)


def evaluate(args, config, store):
    from ..evaluation import evaluate

    return evaluate(config, store, args.run)


def report(args, config, store):
    from ..evaluation import report

    return report(config, store, args.run)


def usage(args, config, store):
    from ..api import usage_summary

    calls = [
        {"context": r[0], "role": r[1], **json.loads(r[2])}
        for r in store.db.execute("SELECT context,role,payload FROM api_calls ORDER BY id")
    ]
    path = config.work_dir / "api-usage.json"
    summary = usage_summary(calls)
    write_json(path, {"calls": calls, "summary": summary})
    return {"path": str(path), "summary": summary}
