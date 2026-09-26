"""Reproducible reviewed fact releases; no chat, embedding or benchmark access."""

import argparse
import json
import shutil
import sqlite3
import zlib
from pathlib import Path

from .common import RagError, canonical, digest, read_json, sha256, write_json
from .storage import Store

CONTRACT = "reviewed-financial-facts-v2"
RISK_CHECKS = {
    "table_region_numbers_not_assigned",
    "table_region_text_not_assigned",
    "empty_table_needs_image_route",
}


def prepare(source_work, workspace):
    from .diagnostics.parsing_quality import inspect_snapshot

    sources = []
    with Store(source_work, read_only=True) as store:
        for row in store.db.execute("SELECT id FROM documents ORDER BY id"):
            doc = store.get("documents", row[0])
            pid = store.meta("rawparse:structured:" + doc["id"])
            if not pid:
                raise RagError("Raw structured parse missing for " + doc["doc_id"])
            raw = store.get("parses", pid)
            if (
                sha256(Path(doc["pdf_path"])) != doc["sha256"]
                or sha256(Path(raw["snapshot"])) != raw["snapshot_sha256"]
            ):
                raise RagError("Source hash mismatch.")
            sources.append({"document": doc, "raw": raw})
            path = workspace / "diagnostics" / (doc["doc_id"] + ".json")
            cached = read_json(path) if path.exists() else {}
            if (
                cached.get("raw_snapshot_sha256") != raw["snapshot_sha256"]
                or cached.get("pdf_sha256") != doc["sha256"]
            ):
                diagnostic = inspect_snapshot(read_json(Path(raw["snapshot"])), doc["pdf_path"])
                diagnostic.update(raw_snapshot_sha256=raw["snapshot_sha256"], pdf_sha256=doc["sha256"])
                write_json(path, diagnostic)
    write_json(workspace / "sources.json", sources)
    return {"documents": len(sources), "pages": sum(x["document"]["pages"] for x in sources)}


def risk_pages(source, diagnostic, profile, context=1):
    excluded = {p["page_number"] for p in profile["excluded_pages"]}
    core = {
        p for i in diagnostic["issues"] if i["check"] in RISK_CHECKS for p in i.get("pages", [])
    } - excluded
    pages = {
        q for p in core for q in range(max(1, p - context), min(source["document"]["pages"], p + context) + 1)
    } - excluded
    return sorted(core), sorted(pages)


def cloud_prepare(workspace, profiles, max_pages=400):
    from .mineru_packets import prepare_document

    pending = []
    for source in read_json(workspace / "sources.json"):
        doc = source["document"]
        name = doc["doc_id"]
        cache = workspace / "cloud" / name / "converted/document.json"
        if cache.exists():
            continue
        profile = read_json(profiles / (name + ".json"))
        core, pages = risk_pages(source, read_json(workspace / "diagnostics" / (name + ".json")), profile)
        if pages:
            pending.append((doc, profile, core, pages))
    total = sum(len(x[3]) for x in pending)
    if total > max_pages:
        raise RagError(f"Risk-page submission requires {total} pages, above declared cap {max_pages}.")
    for doc, profile, core, pages in pending:
        manifest = prepare_document(
            doc,
            {"doc_id": doc["doc_id"], "pdf_sha256": doc["sha256"], "pages": profile["excluded_pages"]},
            workspace / "cloud" / doc["doc_id"],
            overlap=0,
            selected_pages=pages,
        )
        manifest["risk_core_pages"] = core
        manifest["purpose"] = "Selective repairs for reviewed six-document fact layer"
        write_json(workspace / "cloud" / doc["doc_id"] / "manifest.json", manifest)
    return {"new_pages": total, "documents": len(pending), "page_cap": max_pages}


def cloud_run(workspace, profiles, action):
    from .mineru_cloud import fetch_trial
    from .mineru_layout_bridge import convert_document
    from .mineru_packets import submit_document

    result = []
    for source in read_json(workspace / "sources.json"):
        name = source["document"]["doc_id"]
        bundle = workspace / "cloud" / name
        if (bundle / "converted/document.json").exists() or not (bundle / "manifest.json").exists():
            continue
        if action == "submit":
            if (bundle / "job.json").exists():
                result.append({"doc_id": name, "state": "already_reserved"})
                continue
            job = submit_document(bundle)
            result.append({"doc_id": name, "state": job["state"]})
        else:
            state = fetch_trial(bundle)
            if state["complete"] and all(x["state"] == "done" for x in state["results"]):
                convert_document(
                    bundle,
                    source["document"],
                    bundle / "converted",
                    read_json(profiles / (name + ".json"))["boundaries"],
                )
                write_json(
                    bundle / "converted/receipt.json",
                    {
                        "source_pdf_sha256": source["document"]["sha256"],
                        "snapshot_sha256": sha256(bundle / "converted/document.json"),
                        "origin": "selective risk pages with one-page context",
                    },
                )
            result.append(
                {
                    "doc_id": name,
                    "complete": state["complete"],
                    "states": [x["state"] for x in state["results"]],
                }
            )
    return result


def _copy_file(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
    if sha256(Path(source)) != sha256(dest):
        raise RagError("Existing release resource differs; use a fresh release directory.")


def _resource(store, locator, value):
    path = store.root / locator
    write_json(path, value)
    with store.db:
        store.db.execute(
            "INSERT OR REPLACE INTO resources VALUES (?,?,?)", (locator, sha256(path), path.read_bytes())
        )


def _summary(report, profile):
    return {
        k: report[k]
        for k in [
            "doc_id",
            "parse_id",
            "raw_parse_id",
            "source_pages",
            "excluded_pages",
            "nodes",
            "tables",
            "pictures",
            "status",
        ]
    } | {
        "table_candidates": len(report["table_repairs"]),
        "tables_replaced": sum(d["accepted"] for d in report["table_repairs"]),
        "spans_repaired": sum(s["accepted"] for s in report["span_repairs"]),
        "assets_reclassified": len(report.get("asset_reclassifications", [])),
        "quality_flags": report["quality"]["counts"],
        "reviewed_pages": profile["reviewed_pages"],
        "outline_exception_groups": len(profile.get("unresolved", [])),
    }


def build(workspace, profiles, output, config):
    from .asset_roles import empty_tables_as_pictures
    from .cleaning_profiles import apply_profile
    from .diagnostics.parsing_quality import inspect_snapshot
    from .parsing import docling_nodes, load_docling
    from .sources import Reader, validate_citations
    from .table_repairs import repair_invalid_spans, repair_tables

    sources = read_json(workspace / "sources.json")
    identity = digest(
        [
            CONTRACT,
            [
                {
                    "raw": s["raw"]["snapshot_sha256"],
                    "profile": read_json(profiles / (s["document"]["doc_id"] + ".json")),
                    "cloud": sha256(workspace / "cloud" / s["document"]["doc_id"] / "converted/document.json")
                    if (workspace / "cloud" / s["document"]["doc_id"] / "converted/document.json").exists()
                    else None,
                }
                for s in sources
            ],
        ]
    )
    rows = []
    with Store(output) as store:
        if store.meta("build_identity", identity) != identity:
            raise RagError("Release inputs changed; build into a new output directory.")
        store.set_meta("build_identity", identity)
        for source in sources:
            doc = dict(source["document"])
            raw = source["raw"]
            name = doc["doc_id"]
            print("Building facts: " + name, flush=True)
            profile = read_json(profiles / (name + ".json"))
            if (
                profile["doc_id"] != name
                or profile["pdf_sha256"] != sha256(Path(doc["pdf_path"]))
                or profile["raw_snapshot_sha256"] != sha256(Path(raw["snapshot"]))
            ):
                raise RagError("Review profile is bound to a different source version.")
            ready_path = store.root / f"cleaning/{name}/report.json"
            if ready_path.exists():
                cached = read_json(ready_path)
                parsed = store.get("parses", cached["parse_id"])
                frozen_profile = read_json(store.root / f"cleaning/{name}/profile.json")
                if (
                    frozen_profile == profile
                    and sha256(Path(parsed["snapshot"])) == parsed["snapshot_sha256"]
                ):
                    rows.append(_summary(cached, profile))
                    continue
            pdf_dest = store.root / "raw" / doc["filename"]
            _copy_file(doc["pdf_path"], pdf_dest)
            doc["pdf_path"] = str(pdf_dest)
            store.put("documents", doc["id"], doc)
            raw_dest = store.root / "parses" / raw["id"] / "document.json"
            _copy_file(raw["snapshot"], raw_dest)
            raw_record = {
                k: raw[k]
                for k in ("id", "doc_version", "kind", "status", "snapshot_sha256", "semantic_signature")
                if k in raw
            }
            raw_record["snapshot"] = str(raw_dest)
            original = load_docling(raw_record)
            store.put_parse(raw_record, docling_nodes(original, doc, raw["id"]))
            store.save_snapshot(raw_dest, raw["snapshot_sha256"])
            store.set_meta("rawparse:structured:" + doc["id"], raw["id"])
            candidate, structure = apply_profile(original, profile)
            raw_diag = read_json(workspace / "diagnostics" / (name + ".json"))
            mapping = structure["exclusions"]["old_to_new_refs"]
            suspects = {
                mapping[x["ref"]]
                for x in raw_diag["issues"]
                if x["check"] in RISK_CHECKS and x["ref"] in mapping
            }
            cloud = workspace / "cloud" / name / "converted/document.json"
            decisions = []
            cloud_hash = None
            if cloud.exists():
                receipt = read_json(cloud.parent / "receipt.json")
                if receipt["source_pdf_sha256"] != doc["sha256"] or receipt["snapshot_sha256"] != sha256(
                    cloud
                ):
                    raise RagError("Cloud candidate receipt/source mismatch.")
                cloud_doc = load_docling({"snapshot": str(cloud)})
                cloud_hash = sha256(cloud)
                candidate, decisions = repair_tables(candidate, cloud_doc, suspects, doc["pdf_path"])
                _resource(
                    store,
                    f"cleaning/{name}/cloud-tables.json",
                    {
                        "sha256": cloud_hash,
                        "tables": [t.model_dump(mode="json") for t in cloud_doc.tables],
                        "source_pdf_sha256": doc["sha256"],
                    },
                )
            else:
                decisions = [
                    {"ref": r, "accepted": False, "reason": "cloud_candidate_unavailable"}
                    for r in sorted(suspects)
                ]
            candidate, spans = repair_invalid_spans(candidate, doc["pdf_path"])
            asset_refs = [mapping[r] for r in profile.get("empty_tables_as_pictures", [])]
            candidate, asset_map, asset_decisions = empty_tables_as_pictures(candidate, asset_refs)
            pid = digest([CONTRACT, raw["snapshot_sha256"], profile, cloud_hash])
            dest = store.root / "parses" / pid / "document.json"
            snapshot = candidate.model_dump(mode="json")
            write_json(dest, snapshot)
            parsed = {
                "id": pid,
                "doc_version": doc["id"],
                "kind": "structured",
                "status": "success",
                "snapshot": str(dest),
                "snapshot_sha256": sha256(dest),
                "raw_parse_id": raw["id"],
                "cleaning_contract": CONTRACT,
                "profile_sha256": digest(profile),
                "cloud_sha256": cloud_hash,
            }
            nodes = docling_nodes(candidate, doc, pid)
            store.put_parse(parsed, nodes)
            store.save_snapshot(dest, parsed["snapshot_sha256"])
            store.set_meta("parse:structured:" + doc["id"], pid)
            excluded = {x["page_number"] for x in profile["excluded_pages"]}
            errors = []
            for n in nodes:
                for s in n["sources"]:
                    if s["page_number"] in excluded or s["page_number"] != s["page_index"] + 1:
                        errors.append({"ref": n["ref"], "source": s})
                    if s.get("bbox"):
                        x0, y0, x1, y1 = s["bbox"]
                        size = candidate.pages[s["page_number"]].size
                        if not (0 <= x0 < x1 <= size.width + 0.1 and 0 <= y0 < y1 <= size.height + 0.1):
                            errors.append({"ref": n["ref"], "issue": "invalid_bbox"})
            if errors:
                raise RagError("Invalid cleaned source geometry/page association: " + str(errors[:2]))
            quality = inspect_snapshot(snapshot, doc["pdf_path"])
            index = {"id": digest([pid, "readback"]), "kind": "structured", "parse_ids": [pid]}
            body = [n for n in nodes if n["reading_order"] is not None and n["sources"]]
            probes = []
            choices = [body[0], body[len(body) // 2], body[-1]]
            choices.extend(
                next((n for n in body if n["kind"] == kind), body[0]) for kind in ["table", "picture"]
            )
            for n in choices:
                reader = Reader(config, store, index)
                result = reader.read(node_id=n["id"], include_images=True)
                if (
                    "error" in result
                    or validate_citations(list(reader.evidence), reader.public_evidence())["validity"] != 1
                ):
                    raise RagError("Cleaned source readback failed.")
                probes.append(
                    {
                        "ref": n["ref"],
                        "pages": [s["page_number"] for s in n["sources"]],
                        "text_tokens": reader.text_used,
                        "images": len(reader.images),
                        "passed": True,
                    }
                )
            report = {
                "doc_id": name,
                "parse_id": pid,
                "raw_parse_id": raw["id"],
                "source_pages": doc["pages"],
                "excluded_pages": sorted(excluded),
                "nodes": len(nodes),
                "tables": len(candidate.tables),
                "pictures": len(candidate.pictures),
                "structure": structure,
                "table_repairs": decisions,
                "span_repairs": spans,
                "asset_reclassifications": asset_decisions,
                "post_repair_ref_map": asset_map,
                "quality": quality,
                "outline_unresolved": profile.get("unresolved", []),
                "readback": probes,
                "source_errors": errors,
                "status": "validated_with_exceptions",
            }
            _resource(store, f"cleaning/{name}/profile.json", profile)
            _resource(store, f"cleaning/{name}/report.json", report)
            rows.append(_summary(report, profile))
        fk = store.db.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RagError("Fact release has foreign key violations.")
        manifest = {
            "contract": CONTRACT,
            "release_id": digest([r["parse_id"] for r in rows]),
            "documents": rows,
            "foreign_key_errors": 0,
            "semantic_status": "pending_final_heading_tree_review",
            "publication_status": "candidate",
            "requires_final_outline_review": True,
            "embedding_calls": 0,
            "chat_api_calls": 0,
            "active_retrieval_indices": False,
        }
        _resource(store, "release.json", manifest)
        store.set_meta("fact_release", manifest)
    return manifest


def export_release(output, destination):
    with Store(output, read_only=True) as store:
        if not store.meta("fact_release"):
            raise RagError("No completed fact release.")
        if store.meta("fact_release").get("requires_final_outline_review"):
            from .final_outline import verify_stored_acceptance

            verify_stored_acceptance(store)
        if destination.exists():
            raise RagError("Export destination already exists; choose a new path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as target:
            store.db.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")
            target.execute("VACUUM")
    return {"database": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}


def check_release(output, config, chunks=True):
    from .chunking import hybrid_chunks
    from .sources import Reader, validate_citations
    from .tokenization import Tokenizer

    settings = config.model_copy(deep=True)
    settings.chunking.max_tokens = 800
    settings.chunking.table_headers = "multirow"
    settings.chunking.source_scope = "fragment"
    tokenizer = Tokenizer(settings.tokenizer)
    rows = []
    with Store(output, read_only=True) as store:
        release = store.meta("fact_release")
        if not release:
            raise RagError("Release build has not completed.")
        if (
            store.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or store.db.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise RagError("SQLite integrity check failed.")
        for row in release["documents"]:
            print("Checking facts: " + row["doc_id"], flush=True)
            parsed = store.get("parses", row["parse_id"])
            original = store.get("parses", row["raw_parse_id"])
            doc = store.get("documents", parsed["doc_version"])
            for path, checksum in [
                (doc["pdf_path"], doc["sha256"]),
                (parsed["snapshot"], parsed["snapshot_sha256"]),
                (original["snapshot"], original["snapshot_sha256"]),
            ]:
                if sha256(Path(path)) != checksum:
                    raise RagError("Fact source/snapshot hash mismatch.")
            nodes = store.nodes(parsed["id"])
            readable = [n for n in nodes if n["sources"] and n["reading_order"] is not None]
            reader = Reader(
                settings,
                store,
                {"id": digest([parsed["id"], "check"]), "kind": "structured", "parse_ids": [parsed["id"]]},
            )
            result = reader.read(node_id=readable[len(readable) // 2]["id"], include_images=True)
            if (
                "error" in result
                or validate_citations(list(reader.evidence), reader.public_evidence())["validity"] != 1
            ):
                raise RagError("Release source readback failed.")
            detail = {
                "doc_id": row["doc_id"],
                "nodes": len(nodes),
                "source_hashes_valid": True,
                "readback_valid": True,
            }
            if chunks:
                pieces = hybrid_chunks(parsed, nodes, settings, tokenizer)
                if any(
                    s["page_number"] in row["excluded_pages"] or s["page_number"] != s["page_index"] + 1
                    for c in pieces
                    for s in c["sources"]
                ):
                    raise RagError("Chunk source maps an excluded or incorrect physical page.")
                detail.update(
                    chunks=len(pieces),
                    max_tokens=max(tokenizer.count(c["text"]) for c in pieces),
                    token_overflow=sum(c.get("token_overflow", False) for c in pieces),
                    unresolved_fragment_sources=sum(
                        c.get("source_alignment")
                        in {
                            "invalid_or_missing_original_spans",
                            "not_uniquely_aligned",
                            "unlocated_characters",
                        }
                        for c in pieces
                    ),
                )
                write_json(output / "diagnostics" / row["doc_id"] / "chunks.json", pieces)
            rows.append(detail)
    result = {
        "release_id": release["release_id"],
        "documents": rows,
        "integrity": "ok",
        "foreign_key_errors": 0,
        "chunked": chunks,
        "embedding_calls": 0,
    }
    write_json(output / ("acceptance.json" if chunks else "restore-check.json"), result)
    return result


def restore_release(database, output, pdf_dir):
    if (output / "experiment.sqlite").exists():
        raise RagError("Restore requires a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(database, output / "experiment.sqlite")
    with Store(output) as store:
        for row in store.db.execute("SELECT payload FROM parses"):
            record = json.loads(row[0])
            dest = store.resource(record["snapshot"])
            blob = store.db.execute(
                "SELECT data FROM snapshots WHERE sha256=?", (record["snapshot_sha256"],)
            ).fetchone()
            if not blob:
                raise RagError("Release is missing a parse snapshot.")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zlib.decompress(blob[0]))
            if sha256(dest) != record["snapshot_sha256"]:
                raise RagError("Restored snapshot hash mismatch.")
        for row in store.db.execute("SELECT locator,sha256,data FROM resources"):
            dest = store.resource(row[0])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(row[2])
            if sha256(dest) != row[1]:
                raise RagError("Restored resource hash mismatch.")
        for row in store.db.execute("SELECT id FROM documents"):
            doc = store.get("documents", row[0])
            source = pdf_dir / doc["filename"]
            if sha256(source) != doc["sha256"]:
                raise RagError("Restore PDF differs from benchmark source.")
            _copy_file(source, Path(doc["pdf_path"]))
        if store.db.execute("PRAGMA foreign_key_check").fetchall():
            raise RagError("Restored fact relations fail.")
        return store.meta("fact_release")


def main():
    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "cloud-prepare",
            "cloud-submit",
            "cloud-fetch",
            "build",
            "check",
            "export",
            "restore",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--source-work-dir", type=Path, help="Required for prepare: raw input work directory")
    parser.add_argument("--workspace", type=Path, default=Path(".local/cleaning-v1"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles/cleaning-v2"))
    parser.add_argument("--output", type=Path, default=Path(".local/facts-v2-release"))
    parser.add_argument("--database", type=Path, default=Path("handoff/finance-facts-v2.sqlite"))
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--max-pages", type=int, default=400)
    parser.add_argument("--no-chunks", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "prepare":
        if args.source_work_dir is None:
            parser.error("--source-work-dir is required for prepare")
        result = prepare(args.source_work_dir, args.workspace)
    elif args.action == "cloud-prepare":
        result = cloud_prepare(args.workspace, args.profiles, args.max_pages)
    elif args.action in {"cloud-submit", "cloud-fetch"}:
        result = cloud_run(args.workspace, args.profiles, args.action.split("-")[1])
    elif args.action == "build":
        result = build(args.workspace, args.profiles, args.output, config)
    elif args.action == "export":
        result = export_release(args.output, args.database)
    elif args.action == "check":
        result = check_release(args.output, config, chunks=not args.no_chunks)
    else:
        if args.pdf_dir is None:
            parser.error("--pdf-dir is required for restore")
        result = restore_release(args.database, args.output, args.pdf_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
