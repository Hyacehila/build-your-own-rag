"""Build retrieval over an accepted fact release without loading benchmark annotations."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from financial_rag.api import ModelAPI, embedding_key, usage_summary
from financial_rag.common import RagError, canonical, digest, write_json
from financial_rag.config import load_config
from financial_rag.dataset import dataset
from financial_rag.document.cleaning import restore_release
from financial_rag.document.facts import BODY_KINDS
from financial_rag.document.final_outline import verify_stored_acceptance
from financial_rag.indexing.chunking import current_index
from financial_rag.query.sources import Reader
from financial_rag.release_security import secret_scan
from financial_rag.storage import Store
from financial_rag.tokenization import Tokenizer


def attach(store, source_root):
    """Attach reviewed facts to an existing dataset without replacing its questions or runs."""
    from financial_rag.document.parsing import docling_nodes, load_docling

    source_root = Path(source_root).resolve()
    if source_root == store.root:
        raise RagError("Choose a separate reviewed fact workspace.")
    manifest = dataset(store, verify_pdfs=True, verify_queries=True)
    target_docs = {doc["id"]: doc for doc in manifest["documents"]}
    with Store(source_root, read_only=True) as source:
        release = source.meta("fact_release")
        if not release:
            raise RagError("The source workspace has no fact release.")
        acceptance = verify_stored_acceptance(source)
        existing = store.meta("fact_release")
        if existing:
            if existing["release_id"] != release["release_id"]:
                raise RagError("A different fact release is already active; use a new workspace.")
            verify_stored_acceptance(store)
            return {"fact_release_id": release["release_id"], "documents": len(target_docs), "already_attached": True}

        records = []
        versions_by_doc = {}
        seen = set()
        for row in release["documents"]:
            cleaned = source.get("parses", row["parse_id"])
            version = cleaned["doc_version"]
            doc = target_docs.get(version)
            if not doc or version in seen:
                raise RagError("Reviewed documents do not match the prepared dataset.")
            original = source.get("documents", version)
            if any(original.get(k) != doc.get(k) for k in ("id", "doc_id", "filename", "pages")):
                raise RagError("Reviewed document identity or page count differs from the dataset.")
            if (row["doc_id"] != doc["doc_id"] or row["source_pages"] != doc["pages"]
                    or cleaned.get("cleaning_contract") != release["contract"]
                    or cleaned.get("status") != "success"):
                raise RagError("Reviewed parse does not belong to the accepted document.")
            raw = source.get("parses", row["raw_parse_id"])
            if raw.get("doc_version") != version or cleaned.get("raw_parse_id") != raw["id"]:
                raise RagError("Reviewed parse is not linked to its original document.")
            for parsed in (raw, cleaned):
                if not Path(parsed["snapshot"]).is_file():
                    raise RagError("A reviewed parse snapshot is missing.")
                # Fail before changing active pointers if a snapshot is unreadable.
                document = load_docling(parsed)
                nodes = source.nodes(parsed["id"])
                for node in nodes:
                    if node["parse_id"] != parsed["id"]:
                        raise RagError("A reviewed node belongs to another parse.")
                    for location in node.get("sources", []):
                        page = location.get("page_index")
                        if (location.get("doc_version") != version or location.get("doc_id") != doc["doc_id"]
                                or type(page) is not int or not 0 <= page < doc["pages"]
                                or location.get("page_number") != page + 1):
                            raise RagError("A reviewed node has an invalid document or physical page.")
                if docling_nodes(document, original, parsed["id"]) != nodes:
                    raise RagError("Reviewed snapshot and stored document structure differ.")
                old = store.db.execute("SELECT 1 FROM parses WHERE id=?", (parsed["id"],)).fetchone()
                if old:
                    current = store.get("parses", parsed["id"])
                    if (store.nodes(parsed["id"]) != nodes
                            or docling_nodes(load_docling(current), doc, parsed["id"]) != nodes):
                        raise RagError("A parse ID already exists with different document content.")
                records.append((parsed, nodes, bool(old)))
            seen.add(version)
            versions_by_doc[doc["doc_id"]] = version
        if seen != set(target_docs):
            raise RagError("Reviewed documents do not cover the prepared dataset.")

        resources = [tuple(r) for r in source.db.execute("SELECT locator,sha256,data FROM resources")]
        for locator, _, data in resources:
            old = store.db.execute("SELECT data FROM resources WHERE locator=?", (locator,)).fetchone()
            dest = store.resource(locator)
            if (old and old[0] != data) or (dest.exists() and dest.read_bytes() != data):
                raise RagError("A review resource already exists with different content.")

        for parsed, nodes, exists in records:
            if exists:
                continue
            src = Path(parsed["snapshot"])
            dest = store.resource(f"parses/{parsed['id']}/document.json")
            if dest.exists():
                if json.loads(dest.read_text(encoding="utf-8")) != json.loads(src.read_text(encoding="utf-8")):
                    raise RagError("A parse snapshot already exists with different content.")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            parsed["snapshot"] = str(dest)
            store.save_snapshot(dest, parsed["snapshot_sha256"])
            store.put_parse(parsed, nodes)
        for locator, _, data in resources:
            dest = store.resource(locator)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)

        # Switch all documents together; a failed acceptance check rolls back the switch.
        with store.db:
            store.db.executemany("INSERT OR IGNORE INTO resources VALUES (?,?,?)", resources)
            metadata = [("fact_release", release), ("outline_acceptance", acceptance)]
            metadata.extend(
                (f"parse:structured:{versions_by_doc[row['doc_id']]}", row["parse_id"])
                for row in release["documents"]
            )
            store.db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                [(key, canonical(store.portable(value))) for key, value in metadata],
            )
            verify_stored_acceptance(store)
    return {"fact_release_id": release["release_id"], "documents": len(seen), "already_attached": False}


def prepare(config, database, pdf_dir):
    """Copy the accepted release once; restarting never overwrites a working database."""
    sidecar = database.with_suffix(".json")
    if sidecar.exists():
        _published = json.loads(sidecar.read_text(encoding="utf-8"))
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as source:
        row = source.execute("SELECT value FROM metadata WHERE key='fact_release'").fetchone()
        if not row:
            raise RagError("Source has no fact release.")
        expected = json.loads(row[0])["release_id"]
    if not (config.work_dir / "experiment.sqlite").exists():
        restore_release(database, config.work_dir, pdf_dir)
    with Store(config.work_dir) as store:
        if store.meta("fact_release", {}).get("release_id") != expected:
            raise RagError("Target contains a different fact release; use a fresh work_dir.")
        acceptance = verify_stored_acceptance(store)
        manifest = dataset(store, verify_pdfs=True)
        return {
            "fact_release_id": expected,
            "documents": len(manifest["documents"]),
            "accepted_documents": len(acceptance["documents"]),
            "neural_parse_calls": 0,
        }


def reuse_embeddings(config, store, source_database):
    """Only transfer vectors proven to match current text AND complete model identity."""
    role = config.models.embedding
    model_id = digest(role.identity())
    wanted = {
        embedding_key(role, c["text"])
        for kind in ("flat", "structured")
        for c in store.chunks(current_index(config, store, kind)["id"])
    }
    copied, present = 0, 0
    with sqlite3.connect(f"{source_database.resolve().as_uri()}?mode=ro", uri=True) as source:
        # Cache keys without a verifiable source input are intentionally not imported.
        rows = source.execute(
            """SELECT json_extract(c.payload,'$.text'),m.embedding_key,e.dimensions,e.vector
            FROM chunks c JOIN chunk_vectors m ON m.chunk_id=c.id
            JOIN embeddings e ON e.key=m.embedding_key WHERE m.model_id=?""",
            (model_id,),
        )
        seen = set()
        with store.db:
            for text, key, dimensions, blob in rows:
                if key not in wanted or key in seen:
                    continue
                seen.add(key)
                if embedding_key(role, text) != key:
                    raise RagError("Historical vector key does not match its text/model.")
                vector = np.frombuffer(blob, dtype="<f4")
                expected = role.request_params.get("dimensions", dimensions)
                if (
                    dimensions != expected
                    or len(vector) != dimensions
                    or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) == 0
                ):
                    raise RagError("Historical vector has invalid dimensions or values.")
                old = store.db.execute(
                    "SELECT dimensions,vector FROM embeddings WHERE key=?", (key,)
                ).fetchone()
                if old:
                    if tuple(old) != (dimensions, blob):
                        raise RagError("Conflicting vectors for an identical embedding key.")
                    present += 1
                else:
                    store.db.execute("INSERT INTO embeddings VALUES (?,?,?)", (key, dimensions, blob))
                    copied += 1
    result = {
        "source_database": str(source_database),
        "model": role.identity(),
        "wanted_unique_inputs": len(wanted),
        "copied": copied,
        "already_present": present,
    }
    store.set_meta("retrieval_cache_import", result)
    return result


def check(config, store, *, require_vectors=True):
    """Full relational/provenance audit plus bounded SQL/NumPy and source-read checks."""
    acceptance = verify_stored_acceptance(store)
    manifest = dataset(store, verify_pdfs=True)
    docs = {d["id"]: d for d in manifest["documents"]}
    exclusions = {r["doc_id"]: set(r["excluded_pages"]) for r in store.meta("fact_release")["documents"]}
    if (
        store.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
        or store.db.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RagError("SQLite integrity or foreign-key check failed.")
    tokenizer = Tokenizer(config.tokenizer)
    model_id = digest(config.models.embedding.identity())
    reports = []
    for kind in ("flat", "structured"):
        index = current_index(config, store, kind)
        chunks = store.chunks(index["id"])
        nodes = {n["id"]: n for pid in index["parse_ids"] for n in store.nodes(pid)}
        fts = store.fts_name(index["id"])
        indexed_text = dict(store.db.execute(f"SELECT id,text FROM {fts}"))
        if indexed_text != {c["id"]: c["text"] for c in chunks}:
            raise RagError("FTS and chunk text differ.")
        linked = {}
        for cid, nid in store.db.execute(
            "SELECT chunk_id,item_id FROM chunk_items WHERE chunk_id IN (SELECT id FROM chunks WHERE index_id=?)",
            (index["id"],),
        ):
            linked.setdefault(cid, set()).add(nid)
        counts, alignment = Counter(), Counter()
        max_tokens = 0
        for c in chunks:
            if not c["text"].strip() or not c["sources"] or not c["node_ids"]:
                raise RagError("Empty or unlocated retrieval chunk.")
            if linked.get(c["id"]) != set(c["node_ids"]) or any(n not in nodes for n in c["node_ids"]):
                raise RagError("Chunk-to-fact mapping differs from its payload.")
            items = [nodes[n] for n in c["node_ids"]]
            if kind == "structured":
                if any(c["headings"] != n["headings"] for n in items if n["kind"] in BODY_KINDS | {"table"}):
                    raise RagError("Retrieval heading context differs from the accepted fact ancestry.")
                body = [n for n in items if n["kind"] in BODY_KINDS]
                if len({(n["parent_id"], n["section_id"]) for n in body}) > 1:
                    raise RagError("Body merge crossed a parent/section boundary.")
            original_sources = {canonical(s) for n in items for s in n["sources"]}
            if any(canonical(s) not in original_sources for s in c["sources"]):
                raise RagError("Chunk source differs from its original fact coordinates.")
            max_tokens = max(max_tokens, tokenizer.count(c["text"]))
            for src in c["sources"]:
                doc = docs[src["doc_version"]]
                if (
                    src["doc_id"] != doc["doc_id"]
                    or not 0 <= src["page_index"] < doc["pages"]
                    or src["page_number"] != src["page_index"] + 1
                ):
                    raise RagError("Invalid physical-page mapping.")
                if kind == "structured" and src["page_number"] in exclusions[doc["doc_id"]]:
                    raise RagError("Excluded contents page reentered the structured index.")
            if c.get("source_alignment") in {
                "not_uniquely_aligned",
                "unlocated_characters",
                "invalid_or_missing_original_spans",
            }:
                raise RagError("A split chunk has unresolved page alignment.")
            counts[c["kind"]] += 1
            alignment[c.get("source_alignment", "node")] += 1
        if max_tokens > config.chunking.max_tokens:
            raise RagError("Chunk input exceeds configured token target; inspect indivisible table rows.")
        item_ids = {nid for c in chunks for nid in c["node_ids"]}
        unindexed_assets = Counter(
            n["kind"] for n in nodes.values() if n["kind"] in {"picture", "table"} and n["id"] not in item_ids
        )
        reads = []
        for doc_id in docs:
            sample = next((c for c in chunks if c["sources"][0]["doc_version"] == doc_id), None)
            if sample is None:
                raise RagError("Document has no retrieval chunks.")
            reader = Reader(config, store, index)
            response = reader.read(chunk_id=sample["id"], include_images=True)
            if response.get("error") or not reader.evidence:
                raise RagError("Original node/page read failed.")
            reads.append(
                {
                    "doc_id": docs[doc_id]["doc_id"],
                    "citations": len(reader.evidence),
                    "tokens": reader.text_used,
                    "page_images": len(reader.images),
                }
            )
        vector_check = None
        if require_vectors:
            store.map_vectors(index["id"], config.models.embedding)
            vectors = [store.vector(embedding_key(config.models.embedding, c["text"])) for c in chunks]
            matrix = np.stack(vectors)
            if matrix.shape[1] != config.models.embedding.request_params.get("dimensions", matrix.shape[1]):
                raise RagError("Index vector dimensions differ from configuration.")
            # Ties are compared by distance threshold, not float32 ordering noise.
            for pos in sorted({0, len(chunks) // 2, len(chunks) - 1}):
                query = matrix[pos]
                distances = 1 - (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
                actual = store.dense(index["id"], model_id, query, 10)
                positions = {c["id"]: i for i, c in enumerate(chunks)}
                if len(actual) != min(10, len(chunks)) or any(
                    distances[positions[r["id"]]] > np.sort(distances)[min(9, len(chunks) - 1)] + 2e-5
                    for r in actual
                ):
                    raise RagError("SQLite cosine ranking differs from the NumPy reference.")
            vector_check = {
                "chunks_mapped": len(chunks),
                "dimensions": matrix.shape[1],
                "sql_numpy_probes": 3,
            }
            from financial_rag.query.retrieval import Search

            # A real indexed text is a cached query vector. This exercises BM25 +
            # dense + RRF without an API key, new payment or fabricated vectors.
            search = Search(config, store, index, ModelAPI(config, store, "retrieval-integrity"))
            hits = search.search(chunks[0]["text"])
            if not hits or not search.trace[0]["sparse"] or not search.trace[0]["dense"]:
                raise RagError("Hybrid retrieval failed to return both retrieval routes.")
            vector_check["hybrid_smoke"] = {
                "query_chunk_id": chunks[0]["id"],
                "sparse_hits": len(search.trace[0]["sparse"]),
                "dense_hits": len(search.trace[0]["dense"]),
                "fused_hits": len(hits),
                "query_vector": "exact cache hit; no API request",
                "top_chunk_id": hits[0]["id"],
            }
        reports.append(
            {
                "kind": kind,
                "index_id": index["id"],
                "chunks": len(chunks),
                "types": dict(counts),
                "max_tokens": max_tokens,
                "source_alignment": dict(alignment),
                "assets_without_text_entry": dict(unindexed_assets),
                "source_reads": reads,
                "vectors": vector_check,
            }
        )
    local_usage = usage_summary(
        [
            {"role": r[0], **json.loads(r[1])}
            for r in store.db.execute("SELECT role,payload FROM api_calls ORDER BY id")
        ]
    )
    report = {
        "status": "passed",
        "scope": "fact-and-retrieval-integrity; not an answer-quality evaluation",
        "fact_release_id": acceptance["release_id"],
        "indices": reports,
        "embedding": config.models.embedding.identity(),
        "tokenizer": config.tokenizer.model_dump(),
        "api_usage": local_usage or store.meta("retrieval_release", {}).get("api_usage", {}),
        "local_api_usage": local_usage,
        "secret_scan": secret_scan(store.db),
    }
    write_json(store.root / "retrieval-check.json", report)
    if require_vectors:
        store.set_meta("retrieval_release", report)
    return report


def export(config, store, destination):
    report = check(config, store)
    if destination.exists():
        raise RagError("Export destination exists; choose a new snapshot filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as target:
        store.db.backup(target)
        target.execute("PRAGMA foreign_keys=ON")
        active = {r["index_id"] for r in report["indices"]}
        for (iid,) in target.execute("SELECT id FROM indices").fetchall():
            if iid not in active:
                target.execute("DELETE FROM chunk_fts WHERE index_id=?", (iid,))
                target.execute("DELETE FROM chunks WHERE index_id=?", (iid,))
                target.execute("DROP TABLE IF EXISTS " + Store.fts_name(iid))
                target.execute("DELETE FROM indices WHERE id=?", (iid,))
                target.execute("DELETE FROM metadata WHERE key LIKE 'index:%' AND value=?", (canonical(iid),))
        target.execute("DELETE FROM embeddings WHERE key NOT IN (SELECT embedding_key FROM chunk_vectors)")
        # Only this offline copy is pruned. The working database retains its audit trail.
        for table in ("api_calls", "cache", "records", "runs"):
            target.execute(f"DELETE FROM {table}")
        target.commit()
        if target.execute("PRAGMA foreign_key_check").fetchall():
            raise RagError("Export relation validation failed.")
        scan = secret_scan(target)
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("VACUUM")
    result = {
        "database": destination.name,
        "bytes": destination.stat().st_size,
        "report": report,
        "secret_scan": scan,
        "benchmark_annotations": "not attached",
        "scope": "accepted raw/clean facts, complete A/B chunks and vectors; no C/D enhancement or evaluations",
    }
    write_json(destination.with_suffix(".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "reuse", "check", "export"])
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--database", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--without-vectors", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action in {"prepare", "reuse", "export"} and args.database is None:
        parser.error("--database is required")
    if args.action == "prepare":
        if args.pdf_dir is None:
            parser.error("--pdf-dir is required")
        result = prepare(config, args.database, args.pdf_dir)
    else:
        with Store(config.work_dir) as store:
            if args.action == "reuse":
                result = reuse_embeddings(config, store, args.database)
            elif args.action == "export":
                result = export(config, store, args.database)
            else:
                result = check(config, store, require_vectors=not args.without_vectors)
    print(canonical(result))


if __name__ == "__main__":
    main()
