"""Local configuration, data and source checks for the financial example."""

from __future__ import annotations

import json
import os
from pathlib import Path

from financial_rag.indexing.chunking import current_index

from .budget import status
from .common import RagError, digest, read_jsonl, write_json
from .config import Config
from .dataset import dataset
from .storage import Store


def doctor(config: Config, store: Store) -> dict:
    from financial_rag.document.parsing import active_parses

    report = {
        "sqlite": store.db.execute("SELECT sqlite_version()").fetchone()[0],
        "sqlite_vec": store.db.execute("SELECT vec_version()").fetchone()[0],
        "foreign_key_errors": len(store.db.execute("PRAGMA foreign_key_check").fetchall()),
        "integrity": store.db.execute("PRAGMA integrity_check").fetchone()[0],
        "budget": status(config),
        "models": {},
        "failures": [],
        "indices": {},
        "dataset_check": "not_prepared",
    }
    for name in ("embedding", "chat", "vision", "judge"):
        role = getattr(config.models, name)
        report["models"][name] = {
            "model": role.model,
            "base_url": role.base_url,
            "configured": role.configured(),
            "credential_present": bool(os.environ.get(role.api_key_env)) if role.api_key_env else None,
        }
    if report["foreign_key_errors"] or report["integrity"] != "ok":
        report["failures"].append("SQLite integrity or foreign-key check failed")
    if store.meta("dataset"):
        try:
            manifest = dataset(store, verify_pdfs=True, verify_queries=True)
            prepared = Path(manifest["prepared_dir"])
            labels_path = prepared / "labels.jsonl"
            questions = read_jsonl(prepared / "queries.jsonl")
            labels = read_jsonl(labels_path)
            qids = [q["query_id"] for q in questions]
            if (
                len(set(qids)) != len(qids)
                or {q["query_id"] for q in labels} != set(qids)
                or len(labels) != len(qids)
            ):
                raise RagError("Questions and labels are not one-to-one.")
            docs = {d["id"]: d for d in manifest["documents"]}
            for label in labels:
                for rel in label["qrels"]:
                    doc = docs.get(rel["doc_version"])
                    if (
                        not doc
                        or doc["doc_id"] != rel["doc_id"]
                        or not 0 <= rel["page_index"] < doc["pages"]
                        or rel["page_number"] != rel["page_index"] + 1
                    ):
                        raise RagError("A relevance annotation has an invalid physical-page binding.")
            report["dataset_check"] = "passed"
            report["dataset"] = {
                "synthetic": manifest["synthetic"],
                "language": manifest["language"],
                **manifest["counts"],
                "qrels": sum(len(r["qrels"]) for r in labels),
            }
            for kind in ("flat", "structured"):
                if store.meta("index:" + kind):
                    index = current_index(config, store, kind)
                    parses = active_parses(config, store, kind)
                    count = store.db.execute(
                        "SELECT COUNT(*) FROM chunk_items ci JOIN chunks c ON c.id=ci.chunk_id WHERE c.index_id=?",
                        (index["id"],),
                    ).fetchone()[0]
                    report["indices"][kind] = {
                        "id": index["id"],
                        "chunks": index["chunk_count"],
                        "parse_versions": len(parses),
                        "source_relations": count,
                    }
                    model_id = digest(config.models.embedding.identity())
                    vectors = store.db.execute(
                        """SELECT DISTINCT e.key,e.dimensions,LENGTH(e.vector) AS bytes,
                        vec_distance_cosine(e.vector,e.vector) AS self_distance
                        FROM chunks c JOIN chunk_vectors m ON m.chunk_id=c.id AND m.model_id=?
                        JOIN embeddings e ON e.key=m.embedding_key WHERE c.index_id=?""",
                        (model_id, index["id"]),
                    ).fetchall()
                    expected = config.models.embedding.request_params.get("dimensions")
                    if any(
                        r["bytes"] != r["dimensions"] * 4
                        or (expected and r["dimensions"] != expected)
                        or r["self_distance"] is None
                        or abs(r["self_distance"]) > 1e-5
                        for r in vectors
                    ):
                        raise RagError("Invalid cached vector length, dimensions, finite values or norm.")
                    mapped = store.db.execute(
                        """SELECT COUNT(*) FROM chunk_vectors m JOIN chunks c ON c.id=m.chunk_id
                        WHERE c.index_id=? AND m.model_id=?""",
                        (index["id"], model_id),
                    ).fetchone()[0]
                    report["indices"][kind].update(
                        distinct_vectors=len(vectors),
                        mapped_chunks=mapped,
                        vector_status="complete" if mapped == index["chunk_count"] else "pending_embedding",
                    )
        except (RagError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            report["dataset_check"] = "failed"
            report["failures"].append(f"{type(error).__name__}: {error}")
    report["budget"] = status(config)
    write_json(store.root / "doctor.json", report)
    return report
