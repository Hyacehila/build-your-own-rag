"""Bounded live A/B/C/D fixture probe plus current-facts A/B question probe.

The fictional fixture exercises all four arms, including DeepSeek asset
enrichment and Agent tool calls. The current v2 facts are probed separately
without attaching benchmark labels or changing the released retrieval database.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from financial_rag.api import ModelAPI
from financial_rag.chunking import chunk, current_index
from financial_rag.common import RagError, read_json, write_json
from financial_rag.config import Role, load_config
from financial_rag.enrichment import enrich
from financial_rag.evaluation import evaluate, report
from financial_rag.fixture import create_fixture
from financial_rag.parsing import parse
from financial_rag.retrieval import embed_index
from financial_rag.runner import QuestionRunner, run
from financial_rag.storage import Store

QUESTION = (
    "Did JPMorganChase execute more than half of its planned $30 billion "
    "stock repurchase program by year-end 2024?"
)


def record_summary(record: dict) -> dict:
    trace = record.get("trace", [])
    citations = record.get("citations", [])
    return {
        "arm": record["arm"],
        "query_id": record["query_id"],
        "status": record["status"],
        "answer": record.get("answer"),
        "citations": citations,
        "citation_validity": record.get("citation_check", {}).get("validity"),
        "searches": sum(t.get("tool") == "search" for t in trace),
        "reads": sum(t.get("tool") == "read" and t.get("status") == "ok" for t in trace),
        "model_calls": record.get("model_calls", 0),
        "tool_calls": record.get("tool_calls", 0),
        "error": record.get("error"),
        "api_usage": record.get("api_usage"),
    }


def fixture_probe(config, run_id: str) -> dict:
    with Store(config.work_dir) as store:
        if not store.meta("dataset"):
            create_fixture(config, store, structured_snapshot=True)
        if not store.meta("index:flat"):
            parse(config, store, "flat")
            chunk(config, store, "flat")
        if not store.meta("index:structured"):
            chunk(config, store, "structured")

        enhancement = enrich(config, store)
        if enhancement["failed"]:
            raise RagError(f"Fixture enrichment failed: {enhancement['failed']}")
        if not store.meta("index:enriched"):
            chunk(config, store, "enriched")
        indices = {}
        for kind in ("flat", "structured", "enriched"):
            embed_index(config, store, kind)
            indices[kind] = current_index(config, store, kind)

        result = run(config, store, run_id, ["A", "B", "C", "D"], limit=1)
        scored = evaluate(config, store, run_id)
        exported = report(config, store, run_id)
        records = [record_summary(r) for r in store.records(run_id)]
        return {
            "synthetic": True,
            "run": result,
            "evaluation": scored,
            "report": exported,
            "enrichment": enhancement,
            "index_chunks": {kind: index["chunk_count"] for kind, index in indices.items()},
            "records": records,
        }


def current_facts_probe(config, output: Path) -> dict:
    result_path = output / "current-v2-results.json"
    results = read_json(result_path) if result_path.exists() else []
    with Store(config.work_dir) as store:
        for arm, kind in (("A", "flat"), ("B", "structured")):
            if any(r["arm"] == arm and r["status"] == "ok" and r["query"] == QUESTION for r in results):
                print(f"current-v2 {arm}: already completed", flush=True)
                continue
            index = current_index(config, store, kind)
            context = f"preflight-current-v2:{arm}"
            worker = QuestionRunner(config, store, index, ModelAPI(config, store, context))
            record = worker.run({"query_id": "repurchase-pilot", "query": QUESTION}, arm)
            results = [r for r in results if r["arm"] != arm]
            results.append(record)
            write_json(result_path, results)
            print(f"current-v2 {arm}: {record['status']}", flush=True)
    return {"question": QUESTION, "records": [record_summary(r) for r in results]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="abcd-live-preflight-20260925")
    parser.add_argument("--batch", default="abcd-live-preflight-20260925")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("--run-id must be 1-80 letters, digits, underscores or hyphens")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.batch):
        parser.error("--batch must be 1-80 letters, digits, underscores or hyphens")
    project = Path(__file__).resolve().parents[1]
    base = load_config(project / "config.toml")
    output = project / ".local" / args.run_id
    output.mkdir(parents=True, exist_ok=True)

    # A new durable request batch keeps this probe separate from prior validation.
    base.validation.batch = args.batch
    base.validation.ledger = project / ".local" / "validation-attempts.sqlite"
    base.validation.deepseek_request_limit = 20
    base.query.direct_read_hits = 5
    base.query.text_tokens = 5000
    base.query.page_images = 2
    base.models.judge = Role()  # This is a chain check, not an answer-quality study.

    fictional = base.model_copy(deep=True)
    fictional.work_dir = output / "fixture"
    fixture_result = fixture_probe(fictional, args.run_id)
    write_json(output / "fixture-summary.json", fixture_result)

    current = base.model_copy(deep=True)
    actual_result = current_facts_probe(current, output)

    summary = {
        "scope": "one fictional question in A/B/C/D; one current-v2 financial question in A/B",
        "deepseek_request_limit": base.validation.deepseek_request_limit,
        "synthetic": fixture_result,
        "current_v2": actual_result,
    }
    write_json(output / "summary.json", summary)
    print(f"Preflight report: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
