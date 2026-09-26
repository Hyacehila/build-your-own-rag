"""Build reviewable local candidates without changing the active facts or vectors."""

from ..common import RagError, digest, write_json
from ..parsing import docling_nodes, load_docling
from .structural_repairs import CONTRACT, native_grid_candidate, repair_headings


def repair_preview(config, store, doc_ids, output):
    from ..chunking import hybrid_chunks
    from ..tokenization import Tokenizer
    from .parsing_quality import inspect_snapshot

    config = config.model_copy(deep=True)
    config.chunking.table_headers = "multirow"
    config.chunking.source_scope = "fragment"
    docs = {d["doc_id"]: d for d in store.meta("dataset")["documents"]}
    if not doc_ids or set(doc_ids) - docs.keys():
        raise RagError("Choose existing document IDs for repair-preview.")
    results = []
    for name in dict.fromkeys(doc_ids):
        doc = docs[name]
        parsed = store.get("parses", store.meta("parse:structured:" + doc["id"]))
        raw = store.get("parses", parsed["raw_parse_id"]) if parsed.get("raw_parse_id") else parsed
        original = load_docling(raw)
        candidate, hierarchy = repair_headings(original, doc["pdf_path"])
        quality = inspect_snapshot(original.model_dump(mode="json"), doc["pdf_path"])
        suspects = {i["ref"] for i in quality["issues"] if i["check"] == "table_region_text_not_assigned"}
        tables = []
        for i, table in enumerate(candidate.tables):
            if table.self_ref in suspects:
                candidate.tables[i], report = native_grid_candidate(table, doc["pdf_path"])
                tables.append(report)
        pid = digest([raw["snapshot_sha256"], CONTRACT])
        dest = output / name / pid
        snapshot = dest / "document.json"
        write_json(snapshot, candidate.model_dump(mode="json"))
        nodes = docling_nodes(candidate, doc, pid)
        write_json(dest / "nodes.json", nodes)
        chunks = hybrid_chunks({"snapshot": str(snapshot)}, nodes, config, Tokenizer(config.tokenizer))
        write_json(dest / "chunks.json", chunks)
        report = {
            "doc_id": name,
            "diagnostic_only": True,
            "contract": CONTRACT,
            "raw_parse_id": raw["id"],
            "snapshot": str(snapshot),
            "hierarchy": hierarchy,
            "native_tables": tables,
            "chunks": len(chunks),
            "overflow_chunks": sum(c.get("token_overflow", False) for c in chunks),
            "localized_fragments": sum(c.get("source_alignment") == "unique_original_text" for c in chunks),
            "active_database_changed": False,
            "api_calls": 0,
        }
        write_json(dest / "report.json", report)
        results.append(
            {k: v for k, v in report.items() if k not in {"hierarchy", "native_tables"}}
            | {
                "heading_changes": len(hierarchy["changes"]),
                "native_table_candidates": sum(r["accepted"] for r in tables),
                "report": str(dest / "report.json"),
            }
        )
    return {"results": results, "published": False}
