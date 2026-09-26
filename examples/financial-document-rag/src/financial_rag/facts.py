"""Immutable raw parses, separately versioned hierarchy, and relational fact projection."""

from __future__ import annotations

import copy
from pathlib import Path

from .common import RagError, digest, sha256, versions, write_json

NORMALIZATION = {"contract": "existing-heading-levels-v1", **versions("docling-core")}
HEADINGS = {"title", "section_header"}
BODY_KINDS = {"text", "paragraph", "list_item", "formula", "code"}
ASSETS = {"table", "picture"}


def relations(nodes: list[dict]) -> list[dict]:
    """Project explicit ancestry without inventing a title or altering raw node data."""
    result = copy.deepcopy(nodes)
    refs = {n["ref"]: n for n in result}
    if len(refs) != len(result):
        raise RagError("Duplicate fact references.")
    chains = {}

    def ancestors(ref, visiting=()):
        if ref in chains:
            return chains[ref]
        if ref in visiting or ref not in refs:
            raise RagError("Cyclic or unresolved fact parent.")
        parent = refs[ref].get("parent_ref")
        chain = [*ancestors(parent, (*visiting, ref)), parent] if parent else []
        chains[ref] = chain
        return chain

    order = 0
    for node in result:
        chain = ancestors(node["ref"])
        parent = refs.get(node.get("parent_ref"))
        node["parent_id"] = parent["id"] if parent else None
        node["depth"] = len(chain)
        siblings = parent.get("children", []) if parent else [node["ref"]]
        if node["ref"] not in siblings:
            raise RagError("Parent/child references disagree.")
        node["sibling_order"] = siblings.index(node["ref"])
        heading_refs = [r for r in [*chain, node["ref"]] if refs[r]["kind"] in HEADINGS]
        node["section_id"] = refs[heading_refs[-1]]["id"] if heading_refs else None
        node["section_ref"] = heading_refs[-1] if heading_refs else None
        node["headings"] = [refs[r]["text"] for r in heading_refs]
        # Containers and content nested inside an asset remain addressable, but are
        # not independent body-reading positions. Furniture never joins a body window.
        readable = bool(node.get("sources")) and not any(
            refs[r]["kind"] in ASSETS or r == "#/furniture" for r in chain
        )
        node["reading_order"] = order if readable else None
        if readable:
            order += 1
        for child in node.get("children", []):
            if child not in refs or refs[child].get("parent_ref") != node["ref"]:
                raise RagError("Unresolved or inconsistent fact child.")
    return result


def fact_content(document: dict) -> dict:
    """Everything except the two fields this normalization is allowed to modify."""

    def strip(value):
        if isinstance(value, list):
            return [strip(v) for v in value]
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in {"parent", "children"}}
        return value

    return strip(document)


def normalize_document(document):
    before = digest(fact_content(document.model_dump(mode="json")))
    normalized = document.model_copy(deep=True)
    normalized._hierarchize()  # Pinned docling-core API, guarded by complete content invariance.
    after = digest(fact_content(normalized.model_dump(mode="json")))
    if before != after:
        raise RagError("Hierarchy normalization altered content or provenance; no facts saved.")
    return normalized, before


def normalized_parse(config, store, raw: dict, doc: dict) -> dict:
    from .parsing import docling_nodes, load_docling

    if raw["status"] != "success":
        raise RagError("Cannot normalize an incomplete raw parse.")
    pid = digest(
        {"raw_parse_id": raw["id"], "raw_sha256": raw["snapshot_sha256"], "normalization": NORMALIZATION}
    )
    found = store.db.execute("SELECT id FROM parses WHERE id=?", (pid,)).fetchone()
    if found:
        parsed = store.get("parses", pid)
        if (
            Path(parsed["snapshot"]).exists()
            and sha256(Path(parsed["snapshot"])) == parsed["snapshot_sha256"]
        ):
            store.set_meta(f"parse:structured:{doc['id']}", pid)
            return parsed
    document, invariant = normalize_document(load_docling(raw))
    path = store.root / "parses" / pid / "document.json"
    write_json(path, document.model_dump(mode="json"))
    parsed = {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"resolved_options", "elapsed_seconds"}}
    parsed.update(
        id=pid,
        raw_parse_id=raw["id"],
        normalization=NORMALIZATION,
        content_invariant_sha256=invariant,
        snapshot=str(path),
        snapshot_sha256=sha256(path),
    )
    store.save_snapshot(Path(raw["snapshot"]), raw["snapshot_sha256"])
    store.save_snapshot(path, parsed["snapshot_sha256"])
    store.put_parse(parsed, docling_nodes(document, doc, pid))
    store.set_meta(f"parse:structured:{doc['id']}", pid)
    return parsed


def migrate(config, store) -> dict:
    """Verify legacy data, then reuse raw snapshots. This never invokes a PDF model/API."""
    from .dataset import dataset, manifest_identity
    from .parsing import docling_nodes, load_docling, parse_signature, signature_matches

    manifest = dataset(store, verify_pdfs=True, verify_queries=True)
    report = {"neural_parse_calls": 0, "documents": [], "legacy_dataset_id": manifest["id"]}
    # Content-derived identity is independent of absolute storage paths.
    manifest["id"] = manifest_identity(manifest)
    store.set_meta("dataset", manifest)
    for doc in manifest["documents"]:
        store.put("documents", doc["id"], doc)
        row = {"doc_id": doc["doc_id"]}
        for kind in ("flat", "structured"):
            active = store.meta(f"parse:{kind}:{doc['id']}")
            if not active:
                raise RagError(f"Missing existing {kind} parse for {doc['doc_id']}; run parse explicitly.")
            parsed = store.get("parses", active)
            raw = store.get("parses", parsed["raw_parse_id"]) if parsed.get("raw_parse_id") else parsed
            path = Path(raw["snapshot"])
            if (
                raw["doc_version"] != doc["id"]
                or raw["status"] != "success"
                or sha256(path) != raw["snapshot_sha256"]
            ):
                raise RagError("Legacy raw parse content/version check failed.")
            signature = parse_signature(config, kind)
            legacy = raw.get("semantic_signature", raw["signature"])
            if "contract" not in legacy:
                legacy = copy.deepcopy(legacy)
                legacy.pop("implementation", None)
                legacy["options"].pop("artifacts_path", None)
                legacy["contract"] = signature["contract"]
            if not signature_matches(legacy, signature) and not (
                manifest["synthetic"] and raw.get("fixture")
            ):
                raise RagError(
                    "Legacy parser options, dependency versions or weights differ; migration refused."
                )
            raw["semantic_signature"] = signature
            if "resolved_options" in raw:
                raw["resolved_options"].pop("artifacts_path", None)
            store.save_snapshot(path, raw["snapshot_sha256"])
            # Legacy projection lacks containers/relations. Re-project without conversion.
            if kind == "structured":
                nodes = docling_nodes(load_docling(raw), doc, raw["id"])
            else:
                nodes = store.nodes(raw["id"])
            store.put_parse(raw, nodes)
            if kind == "structured":
                store.set_meta(f"rawparse:structured:{doc['id']}", raw["id"])
                normalized = normalized_parse(config, store, raw, doc)
                row.update(
                    raw_parse_id=raw["id"],
                    normalized_parse_id=normalized["id"],
                    content_invariant_sha256=normalized["content_invariant_sha256"],
                )
            row[kind + "_nodes"] = len(nodes)
        report["documents"].append(row)
        print(f"Migrated existing facts: {doc['doc_id']}", flush=True)
    store.set_meta("schema_version", 2)
    report["dataset_id"] = manifest["id"]
    report["foreign_key_errors"] = len(store.db.execute("PRAGMA foreign_key_check").fetchall())
    write_json(store.root / "migration.json", report)
    return report
