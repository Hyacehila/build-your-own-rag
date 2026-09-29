"""Bounded real-PDF diagnostic; partial C/D are explicitly NOT a full benchmark.

Original A/B indices and facts remain unchanged. New summaries, vectors, requests and a
separately named diagnostic index are saved in the working DB; never point at handoff files.
No benchmark label is loaded until all retrieval and answers have completed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from financial_rag.api import ModelAPI, usage_summary
from financial_rag.budget import status
from financial_rag.chunking import current_index
from financial_rag.common import digest, implementation, read_jsonl, sha256, write_json, write_jsonl
from financial_rag.config import load_config
from financial_rag.enrichment import enrich, enriched_chunks
from financial_rag.evaluation import _judge, evidence_coverage, retrieval_metrics
from financial_rag.retrieval import Search
from financial_rag.runner import QuestionRunner
from financial_rag.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--query-ids", nargs="+", default=["0", "1", "44", "180"])
    parser.add_argument("--asset-limit", type=int, default=8)
    parser.add_argument(
        "--picture-limit",
        type=int,
        default=0,
        help="Reserve asset slots for pictures selected from original headings/captions, without qrels",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ledger", type=Path, help="Reuse a diagnostic request ceiling across repair attempts"
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.asset_limit <= 12 or not 1 <= len(args.query_ids) <= 8:
        parser.error("Diagnostic bounds: 0..12 assets, 1..8 queries")
    if not 0 <= args.picture_limit <= args.asset_limit:
        parser.error("picture-limit must be between zero and asset-limit")
    config = load_config(args.config)
    config.validation.ledger = (args.ledger or (args.out / "request-budget.sqlite")).resolve()
    config.validation.batch = "repair-diagnostic"
    config.validation.deepseek_request_limit = 60
    queries = {str(q["query_id"]): q for q in read_jsonl(args.queries)}
    selected = [queries[q] for q in args.query_ids]
    manifest = {
        "scope": "real-PDF diagnostic, C/D partial enhancement; NOT full benchmark",
        "implementation": implementation(),
        "config": config.portable_settings(),
        "queries_sha256": sha256(args.queries),
        "query_ids": args.query_ids,
        "asset_limit": args.asset_limit,
        "picture_limit": args.picture_limit,
        "request_ceiling": 60,
        "request_ledger": str(config.validation.ledger),
        "selection": "explicit diagnostic IDs, not representative; assets selected without answers/qrels",
    }
    if not args.live:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    # Avoid silent overwrite, changing a selection mid-run, or resetting a paid request cap.
    if (args.out / "manifest.json").exists():
        parser.error("Output already contains a run; choose a fresh output directory")
    write_json(args.out / "manifest.json", manifest)
    with Store(config.work_dir) as store:
        first_call_id = store.db.execute("SELECT COALESCE(MAX(id),0) FROM api_calls").fetchone()[0]
        indices = {a: current_index(config, store, k) for a, k in [("A", "flat"), ("B", "structured")]}
        context = "repair-diagnostic:" + digest(str(args.out.resolve()))[:16]
        api = ModelAPI(config, store, context + ":preflight")

        def valid_probe(message):
            if json.loads(message["content"]).get("ok") is not True:
                raise ValueError("Expected JSON ok=true")

        api.validated_chat(
            "chat", [{"role": "user", "content": 'Return only JSON: {"ok": true}'}], valid_probe
        )
        api.embed([q["query"] for q in selected])
        searchers = {a: Search(config, store, idx, api) for a, idx in indices.items()}
        candidates = []
        per_query_assets = []
        for q in selected:
            hits = searchers["B"].search(q["query"])
            per_query_assets.append(list(dict.fromkeys(n for hit in hits for n in hit.get("asset_ids", []))))
        # Round robin, using retrieval only, never reference answers or relevance labels.
        for rank in range(max(map(len, per_query_assets), default=0)):
            for assets in per_query_assets:
                if rank < len(assets) and assets[rank] not in candidates:
                    candidates.append(assets[rank])
        pictures = []
        if args.picture_limit:
            nodes = [
                n
                for pid in indices["B"]["parse_ids"]
                for n in store.nodes(pid)
                if n["kind"] == "picture" and (n["headings"] or n.get("caption"))
            ]
            rankings = []
            for q in selected:
                terms = set(re.findall(r"\w+", q["query"].lower()))

                def score(node):
                    text = " ".join(
                        [
                            *node["headings"],
                            node.get("caption", ""),
                            *[s["doc_id"].replace("_", " ") for s in node["sources"]],
                        ]
                    )
                    return len(terms.intersection(re.findall(r"\w+", text.lower())))

                rankings.append(sorted(nodes, key=lambda n: (-score(n), n["id"])))
            for rank in range(len(nodes)):
                for ranking in rankings:
                    if ranking[rank]["id"] not in pictures:
                        pictures.append(ranking[rank]["id"])
                if len(pictures) >= args.picture_limit:
                    break
            pictures = pictures[: args.picture_limit]
        assets = list(dict.fromkeys([*pictures, *candidates]))[: args.asset_limit]
        write_json(
            args.out / "selected-assets.json",
            {
                "ids": assets,
                "pictures": pictures,
                "selection": "picture slots by original headings/captions/doc names, other slots by B retrieval; no labels",
            },
        )
        enhancement = enrich(config, store, assets)
        write_json(args.out / "enhancement.json", enhancement)
        if enhancement["failed"]:
            raise RuntimeError(
                "Selected enhancement assets failed; diagnostic stopped before answer generation"
            )
        pieces, enhancement_ids = enriched_chunks(config, store, indices["B"], diagnostic_partial=True)
        idx = {
            **indices["B"],
            "kind": "enriched",
            "diagnostic_only": True,
            "enrichment_retrieval_mode": "additive",
            "enhancement_ids": enhancement_ids,
            "chunk_count": len(pieces),
        }
        idx["id"] = digest({**idx, "chunks": pieces})
        for i, piece in enumerate(pieces):
            piece["index_id"] = idx["id"]
            piece["id"] = digest([idx["id"], i, piece])
        old_current = store.meta("index:enriched")
        store.put_index(idx, pieces)
        # put_index sets the current index: do not publish a partial diagnostic as current.
        if old_current:
            store.set_meta("index:enriched", old_current)
        else:
            with store.db:
                store.db.execute("DELETE FROM metadata WHERE key='index:enriched'")
        api.embed([p["text"] for p in pieces])  # cache reuse; only descriptions should be new
        indices.update(C=idx, D=idx)
        searchers["C"] = Search(config, store, idx, api)
        searchers["D"] = searchers["C"]
        records = []
        for q in selected:
            for arm in "ABCD":
                worker = QuestionRunner(
                    config,
                    store,
                    indices[arm],
                    ModelAPI(config, store, context + f":{arm}:{q['query_id']}"),
                    searchers[arm],
                )
                result = worker.run(q, arm)
                records.append(result)
                write_jsonl(args.out / "answers.jsonl", records)
                print(f"{arm} {q['query_id']}: {result['status']}", flush=True)
        if args.labels:
            labels = {str(row["query_id"]): row for row in read_jsonl(args.labels)}
            for row in records:
                label = labels[str(row["query_id"])]
                row["judge"] = _judge(config, store, row, label)
                row["metrics"] = {
                    **retrieval_metrics(row["initial_page_ranking"], label["qrels"]),
                    "evidence_page_coverage": evidence_coverage(row["evidence"], label["qrels"]),
                }
                write_jsonl(args.out / "scored.jsonl", records)
        calls = [
            json.loads(r[0]) | {"role": r[1]}
            for r in store.db.execute(
                "SELECT payload,role FROM api_calls WHERE context LIKE ?", (context + "%",)
            )
        ]
        all_new_calls = [
            json.loads(r[0]) | {"role": r[1]}
            for r in store.db.execute("SELECT payload,role FROM api_calls WHERE id>?", (first_call_id,))
        ]
        write_json(
            args.out / "summary.json",
            {
                **manifest,
                "indices": {a: i["id"] for a, i in indices.items()},
                "enhanced_assets_available": len(enhancement_ids),
                "budget": status(config),
                "answer_and_preflight_usage": usage_summary(calls),
                "all_new_call_usage_including_enrichment_and_judge": usage_summary(all_new_calls),
                "rows": [
                    {
                        "arm": r["arm"],
                        "query_id": r["query_id"],
                        "status": r["status"],
                        "correct": r.get("judge", {}).get("correct"),
                        "metrics": r.get("metrics"),
                        "chat_usage": r.get("api_usage"),
                        "error": r.get("error"),
                    }
                    for r in records
                ],
            },
        )


if __name__ == "__main__":
    main()
