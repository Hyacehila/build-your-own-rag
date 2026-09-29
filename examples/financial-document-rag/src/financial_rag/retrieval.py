from __future__ import annotations

import math
import re
import time
from collections import defaultdict

from .api import ModelAPI, embedding_key, usage_summary
from .chunking import current_index
from .common import RagError, digest
from .config import ARMS, Config
from .parsing import active_parses
from .storage import Store
from .tokenization import Tokenizer


def page_ranking(chunks: list[dict]) -> list[dict]:
    """Best chunk occurrence per page; ties use doc ID then zero-based page index."""
    seen, result = set(), []
    for rank, chunk in enumerate(chunks, 1):
        for source in sorted(chunk["sources"], key=lambda s: (s["doc_id"], s["page_index"])):
            key = (source["doc_id"], source["page_index"])
            if key not in seen:
                seen.add(key)
                result.append(
                    {
                        "doc_id": source["doc_id"],
                        "doc_version": source["doc_version"],
                        "page_index": source["page_index"],
                        "page_number": source["page_number"],
                        "chunk_rank": rank,
                        "score": chunk.get("score"),
                    }
                )
    return result


def embed_index(config: Config, store: Store, kind: str) -> dict:
    index = current_index(config, store, kind)
    context = "embed:" + digest([index["id"], config.models.embedding.identity()])
    texts = [c["text"] for c in store.chunks(index["id"])]
    api = ModelAPI(config, store, context)
    vectors = api.embed(texts)
    store.map_vectors(index["id"], config.models.embedding)
    if len({len(v) for v in vectors}) != 1:
        raise RagError("Index embeddings have inconsistent dimensions; change embedding cache_revision.")
    record = {
        "index_id": index["id"],
        "embedding": config.models.embedding.identity(),
        "dimensions": len(vectors[0]),
        "chunks": len(texts),
        "usage": usage_summary(store.calls(context)),
    }
    store.set_meta("embedded:" + digest([index["id"], config.models.embedding.identity()]), record)
    return record


def estimate(config: Config, store: Store) -> dict:
    from .enrichment import enrichment_key, image_input

    tokenizer = Tokenizer(config.tokenizer)
    seen, report = set(), []
    total_tokens = total_requests = 0
    for kind in dict.fromkeys(ARMS.values()):
        iid = store.meta("index:" + kind)
        if iid:
            index = current_index(config, store, kind)
            texts = [c["text"] for c in store.chunks(index["id"])]
            status = "actual_chunk_text"
        elif kind == "enriched" and store.meta("index:structured"):
            base = current_index(config, store, "structured")
            texts = [c["text"] for c in store.chunks(base["id"]) if c["kind"] not in ("table", "picture")]
            status = "body_text_only_until_enrichment_and_chunking"
        else:
            report.append({"index": kind, "status": "not_chunked"})
            continue
        inputs = {embedding_key(config.models.embedding, text): text for text in texts}
        cached = {key for key in inputs if store.vector(key) is not None}
        missed = set(inputs) - cached
        incremental = missed - seen
        tokens = sum(tokenizer.count(inputs[k]) for k in incremental)
        requests = math.ceil(len(incremental) / config.models.embedding.batch_size)
        report.append(
            {
                "index": kind,
                "arms": [a for a, k in ARMS.items() if k == kind],
                "status": status,
                "chunks": len(texts),
                "max_input_tokens": max(map(tokenizer.count, texts), default=0),
                "input_tokens_with_duplicates": sum(map(tokenizer.count, texts)),
                "distinct_inputs": len(inputs),
                "cache_hits": len(cached),
                "cache_miss_inputs": len(missed),
                "reused_from_prior_indices": len(missed & seen),
                "cache_miss_tokens_before_cross_index_reuse": sum(tokenizer.count(inputs[k]) for k in missed),
                "incremental_inputs": len(incremental),
                "incremental_tokens": tokens,
                "incremental_requests": requests,
                "estimated_cost": tokens / 1_000_000 * config.pricing.embedding_per_million,
            }
        )
        total_tokens += tokens
        total_requests += requests
        seen.update(missed)
    projected = defaultdict(int)
    if not store.meta("index:enriched"):
        try:
            parses = active_parses(config, store, "structured")
        except RagError:
            parses = []
        for p in parses:
            for n in store.nodes(p["id"]):
                if n["kind"] not in ("picture", "table"):
                    continue
                record = store.cached(enrichment_key(config, n))
                if record:
                    projected["generated_asset_text_tokens_not_yet_chunked"] += tokenizer.count(
                        "\n".join([*n["headings"], n.get("caption", ""), record["text"]]).strip()
                    )
                else:
                    kind = (
                        "empty_table_image_fallback" if n["kind"] == "table" and image_input(n) else n["kind"]
                    )
                    projected[kind + "_count"] += 1
                    projected[kind + "_estimated_tokens"] += (
                        config.enrichment.image_estimate_tokens
                        if image_input(n)
                        else config.enrichment.table_estimate_tokens
                    )
    return {
        "tokenizer": config.tokenizer.model_dump(),
        "model_configured": config.models.embedding.configured(),
        "pricing": config.pricing.model_dump(),
        "indices": report,
        "projected_assets_separate": dict(projected),
        "measured_text_estimated_tokens_to_embed": total_tokens,
        "new_requests_estimated": total_requests,
        "estimated_embedding_cost_excluding_projected_assets": total_tokens
        / 1_000_000
        * config.pricing.embedding_per_million,
        "formula": "cache-miss distinct input tokens / 1,000,000 * unit price",
        "note": "No generation/vision/judge prices estimated. Provider usage is recorded during actual calls.",
    }


class Search:
    def __init__(self, config: Config, store: Store, index: dict, api: ModelAPI):
        self.config, self.store, self.index, self.api = config, store, index, api
        self.chunks = store.chunks(index["id"])
        self.by_id = {c["id"]: c for c in self.chunks}
        self.model_id = digest(config.models.embedding.identity())
        store.map_vectors(index["id"], config.models.embedding)
        dims = store.db.execute(
            """SELECT DISTINCT e.dimensions FROM chunks c
            JOIN chunk_vectors m ON m.chunk_id=c.id AND m.model_id=?
            JOIN embeddings e ON e.key=m.embedding_key WHERE c.index_id=?""",
            (self.model_id, index["id"]),
        ).fetchall()
        if len(dims) != 1:
            raise RagError("Empty index or inconsistent vector dimensions.")
        self.dimensions = dims[0][0]
        self.trace = []

    def search(self, query: str) -> list[dict]:
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise RagError("Search query must be nonempty text.")
        cfg = self.config.retrieval
        qv = self.api.embed([query])[0]
        if len(qv) != self.dimensions:
            raise RagError("Query vector dimensions differ from index; check model/cache_revision.")
        dense = [r["id"] for r in self.store.dense(self.index["id"], self.model_id, qv, cfg.dense_k)]
        # Quote individual tokens; never execute user-supplied FTS5 syntax.
        terms = list(dict.fromkeys(re.findall(r"\w+", query.lower())))[:64]
        expression = " OR ".join('"' + t + '"' for t in terms)
        # Separate FTS tables keep BM25 corpus statistics independent across A/B/C.
        fts = self.store.fts_name(self.index["id"])
        sparse = (
            []
            if not expression
            else [
                r[0]
                for r in self.store.db.execute(
                    f"SELECT id FROM {fts} WHERE {fts} MATCH ? ORDER BY bm25({fts}),id LIMIT ?",
                    (expression, cfg.sparse_k),
                )
            ]
        )
        fused = defaultdict(float)
        for ranking in (sparse, dense):
            for rank, cid in enumerate(ranking, 1):
                fused[cid] += 1 / (cfg.rrf_k + rank)
        result = [
            {**self.by_id[cid], "score": score}
            for cid, score in sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        ]
        raw_result = result
        if cfg.group_sources:
            grouped = {}
            for hit in result:
                # Group alternative descriptions/rows of ONE asset, not an entire PDF page.
                # Distinct text fragments remain distinct, including same-node split spans.
                assets = tuple(sorted(hit.get("asset_ids", [])))
                key = ("asset", assets) if assets else ("fragment", hit["id"])
                if key not in grouped:
                    grouped[key] = {**hit, "matched_chunk_ids": []}
                grouped[key]["matched_chunk_ids"].append(hit["id"])
            result = list(grouped.values())
        result = result[: cfg.final_k]
        self.trace.append(
            {
                "tool": "search",
                "query": query,
                "index_id": self.index["id"],
                "sparse": sparse,
                "dense": dense,
                "ranked_chunks": [{"id": c["id"], "score": c["score"]} for c in result],
                "candidate_chunks": len(raw_result),
                "source_groups_returned": len(result),
                "page_ranking": page_ranking(result),
                "seconds": time.perf_counter() - started,
            }
        )
        return result
