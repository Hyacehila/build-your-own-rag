"""Read-only, source-only audit of saved facts and slices. No generation or embedding calls."""

from __future__ import annotations

import csv
import hashlib
import json
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf

from ..chunking import current_index
from ..common import RagError, digest, read_json, sha256, write_json
from ..dataset import dataset
from ..facts import ASSETS, BODY_KINDS, HEADINGS, fact_content
from ..tokenization import Tokenizer
from .text_checks import lexical, numbers, recovery


def missing_header_terms(cells, chunk):
    """Presence check for declared header terms, not header/column correctness."""
    expected = set(lexical("\n".join(c.get("text", "") for c in cells if c.get("column_header"))))
    return sorted(expected - set(lexical(body_text(chunk))))


def distribution(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values)
    return {
        "n": len(values),
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.1)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.9)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in r.items()
                }
                for r in rows
            ]
        )


def snapshot_items(snapshot):
    result = {}
    for value in snapshot.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict) and "self_ref" in item:
                result[item["self_ref"]] = item
    return result


def traversal(items, ref="#/body"):
    output, visiting = [], set()

    def visit(value):
        if value in visiting or value not in items:
            raise RagError("Cyclic, duplicated or unresolved snapshot tree reference.")
        visiting.add(value)
        item = items[value]
        if item.get("prov"):
            output.append(value)
        for child in item.get("children", []):
            visit(child.get("$ref", child.get("cref")))

    visit(ref)
    return output


def body_text(chunk):
    prefix = "\n".join(chunk.get("headings") or [])
    return (
        chunk["text"][len(prefix) :].lstrip("\n")
        if prefix and chunk["text"].startswith(prefix)
        else chunk["text"]
    )


def page_node_text(node, page_number):
    raw = node.get("raw", {})
    # A TableItem can be labelled document_index rather than table.
    if "table_cells" in raw.get("data", {}):
        return "\n".join(c.get("text", "") for c in raw["data"]["table_cells"])
    if node["kind"] == "picture":
        return ""
    located = raw.get("orig", node["text"])
    return "\n".join(
        located[p["charspan"][0] : p["charspan"][1]]
        for p in raw.get("prov", [])
        if p["page_no"] == page_number
    )


def fragment_pages(chunk, nodes):
    """Align body to orig with whitespace removed; never infer ambiguous spans.

    These are fragment pages, excluding repeated ancestor headings. Stored node
    pages deliberately remain unchanged by this diagnostic.
    """
    compact, locations = [], []
    for n in nodes:
        original = n.get("raw", {}).get("orig", n["text"])
        provs = n.get("raw", {}).get("prov", [])
        if any(not 0 <= p["charspan"][0] <= p["charspan"][1] <= len(original) for p in provs):
            return {"status": "invalid_original_charspan"}
        for i, ch in enumerate(original):
            if not ch.isspace():
                compact.append(ch)
                locations.append({p["page_no"] for p in provs if p["charspan"][0] <= i < p["charspan"][1]})
    haystack = "".join(compact)
    needle = "".join(body_text(chunk).split())
    a = haystack.find(needle)
    if not needle or a < 0 or haystack.find(needle, a + 1) >= 0:
        return {"status": "not_uniquely_aligned"}
    covered = locations[a : a + len(needle)]
    if any(not pages for pages in covered):
        return {"status": "unlocated_characters"}
    pages = sorted(set().union(*covered))
    stored = {s["page_number"] for s in chunk["sources"]}
    return {"status": "aligned", "fragment_pages": pages, "extra_node_pages": sorted(stored - set(pages))}


def audit_data(config, store, output: Path, gallery=True):
    if store.db.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise RagError("Data audit requires Store(read_only=True).")
    output.mkdir(parents=True, exist_ok=True)
    api_before = store.db.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    manifest = dataset(store, verify_pdfs=True)
    indices = {kind: current_index(config, store, kind) for kind in ("flat", "structured")}
    chunks = {kind: store.chunks(index["id"]) for kind, index in indices.items()}
    tokenizer = Tokenizer(config.tokenizer)
    issues, page_rows, chunk_rows, table_rows, doc_rows = [], [], [], [], []
    fragment_rows = []
    omitted = []
    details = {}
    samples = []
    by_doc = {kind: defaultdict(list) for kind in indices}
    for kind in indices:
        for c in chunks[kind]:
            versions = {s["doc_version"] for s in c["sources"]}
            if len(versions) != 1:
                issues.append({"severity": "error", "check": "chunk_document_binding", "chunk_id": c["id"]})
            else:
                by_doc[kind][next(iter(versions))].append(c)

    def issue(check, severity="error", **data):
        issues.append({"severity": severity, "check": check, **data})

    for doc in manifest["documents"]:
        print(f"Auditing saved facts and chunks: {doc['doc_id']}", flush=True)
        dv = doc["id"]
        parsed = {k: store.get("parses", store.meta(f"parse:{k}:{dv}")) for k in indices}
        raw = (
            store.get("parses", parsed["structured"]["raw_parse_id"])
            if parsed["structured"].get("raw_parse_id")
            else parsed["structured"]
        )
        snapshots = {}
        for record in {p["id"]: p for p in [*parsed.values(), raw]}.values():
            path = Path(record["snapshot"])
            blob = store.db.execute(
                "SELECT data FROM snapshots WHERE sha256=?", (record["snapshot_sha256"],)
            ).fetchone()
            if sha256(path) != record["snapshot_sha256"]:
                issue("snapshot_file_checksum", doc_id=doc["doc_id"], parse_id=record["id"])
            if blob and hashlib.sha256(zlib.decompress(blob[0])).hexdigest() != record["snapshot_sha256"]:
                issue("embedded_snapshot_checksum", doc_id=doc["doc_id"], parse_id=record["id"])
            snapshots[record["id"]] = read_json(path)
        original = snapshots[raw["id"]]
        normalized = snapshots[parsed["structured"]["id"]]
        invariant = digest(fact_content(original)) == digest(fact_content(normalized))
        original_items, items = snapshot_items(original), snapshot_items(normalized)
        order_preserved = traversal(original_items) == traversal(items)
        if not invariant or not order_preserved:
            issue(
                "normalization_content_or_order",
                doc_id=doc["doc_id"],
                content_equal=invariant,
                order_equal=order_preserved,
            )
        nodes = {k: store.nodes(p["id"]) for k, p in parsed.items()}
        by_id = {n["id"]: n for k in nodes for n in nodes[k]}
        refs = {n["ref"]: n for n in nodes["structured"]}
        mapped_chunks = defaultdict(list)
        for c in by_doc["structured"][dv]:
            for nid in c["node_ids"]:
                mapped_chunks[nid].append(c)
        flat_nodes = sorted(nodes["flat"], key=lambda n: n["sources"][0]["page_index"])
        flat_text, ranges = "", []
        for n in flat_nodes:
            start = len(flat_text)
            flat_text += n["text"] + "\n\n"
            if n["text"].strip():
                ranges.append((start, start + len(n["text"]), n["id"]))
        flat_end = 0
        for c in sorted(by_doc["flat"][dv], key=lambda c: c["char_start"]):
            a, b = c["char_start"], c["char_end"]
            if c["text"] != flat_text[a:b]:
                issue("flat_text_differs_from_source", chunk_id=c["id"])
            if a > flat_end and flat_text[flat_end:a].strip():
                issue("flat_uncovered_text", doc_id=doc["doc_id"], start=flat_end, end=a)
            expected = [nid for begin, end, nid in ranges if begin < b and end > a]
            if expected != c["node_ids"]:
                issue("flat_fragment_page_assignment", chunk_id=c["id"])
            flat_end = max(flat_end, b)
        if flat_text[flat_end:].strip():
            issue("flat_uncovered_tail", doc_id=doc["doc_id"])

        by_page = defaultdict(list)
        charspans_invalid = 0
        multipage_nodes = []
        with pymupdf.open(doc["pdf_path"]) as pdf:
            if len(pdf) != doc["pages"] or {int(k) for k in normalized["pages"]} != set(
                range(1, len(pdf) + 1)
            ):
                issue("document_page_count", doc_id=doc["doc_id"])
            page_text = {i: page.get_text("text", sort=True) for i, page in enumerate(pdf)}
            for n in nodes["structured"]:
                raw_item = items.get(n["ref"])
                if raw_item is None or n["id"] != digest([n["parse_id"], n["ref"]]):
                    issue("node_identity", node_id=n["id"])
                if len({s["page_index"] for s in n["sources"]}) > 1:
                    multipage_nodes.append(n["id"])
                parent = refs.get(n.get("parent_ref"))
                if (parent["id"] if parent else None) != n["parent_id"]:
                    issue("parent_identity", node_id=n["id"])
                if parent and parent["children"][n["sibling_order"]] != n["ref"]:
                    issue("sibling_order", node_id=n["id"])
                ancestors = []
                p = parent
                while p:
                    if p["id"] in ancestors:
                        raise RagError("Cyclic fact ancestry")
                    ancestors.append(p["id"])
                    p = refs.get(p.get("parent_ref"))
                if len(ancestors) != n["depth"]:
                    issue("depth", node_id=n["id"])
                headings = [by_id[x] for x in reversed(ancestors) if by_id[x]["kind"] in HEADINGS]
                if n["kind"] in HEADINGS:
                    headings.append(n)
                if [h["text"] for h in headings] != n["headings"] or (
                    headings[-1]["id"] if headings else None
                ) != n["section_id"]:
                    issue("heading_path", node_id=n["id"])
                provs = n.get("raw", {}).get("prov", [])
                if len(provs) != len(n["sources"]):
                    issue("provenance_count", node_id=n["id"])
                for source, prov in zip(n["sources"], provs):
                    page_index = source["page_index"]
                    if (
                        source["doc_version"] != dv
                        or source["page_number"] != page_index + 1
                        or prov["page_no"] != page_index + 1
                    ):
                        issue("node_page_binding", node_id=n["id"])
                    if not 0 <= page_index < len(pdf):
                        issue("page_outside_document", node_id=n["id"])
                        continue
                    r = pdf[page_index].rect
                    box = prov["bbox"]
                    expected = (
                        [box["l"], r.height - box["t"], box["r"], r.height - box["b"]]
                        if box["coord_origin"] == "BOTTOMLEFT"
                        else [box["l"], box["t"], box["r"], box["b"]]
                    )
                    if not np.allclose(source["bbox"], expected, atol=1e-4, rtol=0):
                        issue("bbox_transform", node_id=n["id"])
                    left, top, right, bottom = source["bbox"]
                    if not (-1 <= left <= right <= r.width + 1 and -1 <= top <= bottom <= r.height + 1):
                        issue("bbox_bounds", node_id=n["id"])
                    a, b = prov.get("charspan", [0, 0])
                    # Docling character spans refer to orig, which can retain a
                    # bullet marker absent from the normalized TextItem.text.
                    located_text = n.get("raw", {}).get("orig", n["text"])
                    if (
                        n["kind"] not in ASSETS
                        and not n.get("raw", {}).get("data")
                        and (a < 0 or a > b or b > len(located_text))
                    ):
                        charspans_invalid += 1
                        issue(
                            "text_charspan_bounds",
                            "warning",
                            doc_id=doc["doc_id"],
                            node_id=n["id"],
                            ref=n["ref"],
                            pages=sorted({s["page_number"] for s in n["sources"]}),
                            span=[a, b],
                            original_length=len(located_text),
                        )
                    by_page[page_index].append(n)
                if (
                    n["text"].strip()
                    and n["kind"] in BODY_KINDS | {"footnote", "caption"}
                    and n["id"] not in mapped_chunks
                ):
                    asset_ancestor = any(by_id[p]["kind"] in ASSETS for p in ancestors)
                    reason = "inside_asset" if asset_ancestor else "not_independently_indexed"
                    omitted.append(
                        {
                            "doc_id": doc["doc_id"],
                            "node_id": n["id"],
                            "ref": n["ref"],
                            "kind": n["kind"],
                            "parent_kind": parent["kind"] if parent else None,
                            "reason": reason,
                            "pages": sorted({s["page_number"] for s in n["sources"]}),
                            "text": n["text"],
                        }
                    )
                    if not asset_ancestor:
                        issue("unindexed_nonempty_body", "warning", node_id=n["id"], doc_id=doc["doc_id"])
                if n["kind"] == "table":
                    cells = n["raw"].get("data", {}).get("table_cells", [])
                    cell_text = "\n".join(c.get("text", "") for c in cells)
                    all_chunks = mapped_chunks[n["id"]]
                    missing_numeric = numbers(cell_text) - numbers("\n".join(c["text"] for c in all_chunks))
                    missing_headers = [
                        {"chunk_id": c["id"], "missing_terms": missing_header_terms(cells, c)}
                        for c in all_chunks
                    ]
                    missing_headers = [r for r in missing_headers if r["missing_terms"]]
                    for missing in missing_headers:
                        issue(
                            "declared_table_header_terms_missing",
                            "warning",
                            doc_id=doc["doc_id"],
                            node_id=n["id"],
                            pages=sorted({s["page_number"] for s in n["sources"]}),
                            **missing,
                        )
                    native = "\n".join(
                        pdf[s["page_index"]].get_text("text", clip=pymupdf.Rect(s["bbox"]), sort=True)
                        for s in n["sources"]
                    )
                    table_rows.append(
                        {
                            "doc_id": doc["doc_id"],
                            "node_id": n["id"],
                            "ref": n["ref"],
                            "pages": sorted({s["page_number"] for s in n["sources"]}),
                            "rows": n["raw"]["data"].get("num_rows"),
                            "columns": n["raw"]["data"].get("num_cols"),
                            "cells": len(cells),
                            "empty": not cell_text.strip(),
                            "header_cells": sum(bool(c.get("column_header")) for c in cells),
                            "chunks": len(all_chunks),
                            "raw_cell_numbers": sum(numbers(cell_text).values()),
                            "chunks_missing_declared_header_terms": len(missing_headers),
                            "missing_cell_numeric_tokens_in_chunks": sum(missing_numeric.values()),
                            "missing_examples": dict(missing_numeric.most_common(10)),
                            "native_crop_numeric_recovery": recovery(numbers(native), numbers(cell_text)),
                            "native_crop_numeric_tokens": sum(numbers(native).values()),
                        }
                    )
                    if missing_numeric:
                        issue(
                            "table_numeric_tokens_absent_from_chunks",
                            "warning",
                            doc_id=doc["doc_id"],
                            node_id=n["id"],
                            missing=dict(missing_numeric.most_common(10)),
                        )
            for p, native in page_text.items():
                page_nodes = {n["id"]: n for n in by_page[p]}.values()
                structured_text = "\n".join(page_node_text(n, p + 1) for n in page_nodes)
                page_rows.append(
                    {
                        "doc_id": doc["doc_id"],
                        "page_index": p,
                        "page_number": p + 1,
                        "pdf_page_label": pdf[p].get_label(),
                        "native_characters": len(native.strip()),
                        "native_lexical_tokens": sum(lexical(native).values()),
                        "docling_characters": len(structured_text.strip()),
                        "native_token_recovery": recovery(lexical(native), lexical(structured_text)),
                        "native_numeric_recovery": recovery(numbers(native), numbers(structured_text)),
                        "text_nodes": sum(n["kind"] in BODY_KINDS for n in page_nodes),
                        "tables": sum(n["kind"] == "table" for n in page_nodes),
                        "pictures": sum(n["kind"] == "picture" for n in page_nodes),
                    }
                )

        asset_positions = {
            n["reading_order"]
            for n in nodes["structured"]
            if n["kind"] in ASSETS and n["reading_order"] is not None
        }
        for kind in indices:
            for c in by_doc[kind][dv]:
                ns = [by_id[nid] for nid in c["node_ids"]]
                expected_sources = {digest(s) for n in ns for s in n["sources"]}
                if {digest(s) for s in c["sources"]} != expected_sources:
                    issue("chunk_node_sources", chunk_id=c["id"])
                mapped = [
                    r[0]
                    for r in store.db.execute(
                        "SELECT item_id FROM chunk_items WHERE chunk_id=? ORDER BY ordinal", (c["id"],)
                    )
                ]
                if mapped != c["node_ids"]:
                    issue("chunk_items_sql_relation", chunk_id=c["id"])
                count = tokenizer.count(c["text"])
                is_body = kind == "structured" and all(n["kind"] in BODY_KINDS for n in ns)
                parents = {n["parent_id"] for n in ns}
                sections = {n["section_id"] for n in ns}
                positions = [n["reading_order"] for n in ns if n["reading_order"] is not None]
                crosses_asset = bool(
                    is_body
                    and positions
                    and any(min(positions) <= p <= max(positions) for p in asset_positions)
                )
                if is_body and (len(parents) > 1 or len(sections) > 1 or crosses_asset):
                    issue(
                        "body_chunk_structure_boundary",
                        "warning",
                        chunk_id=c["id"],
                        parents=len(parents),
                        sections=len(sections),
                        crosses_asset=crosses_asset,
                    )
                if is_body and count > config.chunking.max_tokens:
                    issue("body_token_budget", chunk_id=c["id"], tokens=count)
                if c["kind"] in ASSETS and any(n["kind"] not in ASSETS | {"caption", "footnote"} for n in ns):
                    issue("body_asset_mixture", chunk_id=c["id"])
                split_multipage = kind == "structured" and any(
                    n["id"] in multipage_nodes and len(mapped_chunks[n["id"]]) > 1 for n in ns
                )
                if split_multipage and is_body:
                    alignment = fragment_pages(c, ns)
                    fragment_rows.append(
                        {
                            "chunk_id": c["id"],
                            "doc_id": doc["doc_id"],
                            "stored_pages": sorted({s["page_number"] for s in c["sources"]}),
                            **alignment,
                        }
                    )
                    if alignment.get("extra_node_pages"):
                        issue(
                            "fragment_page_overattribution",
                            "warning",
                            chunk_id=c["id"],
                            doc_id=doc["doc_id"],
                            **alignment,
                        )
                chunk_rows.append(
                    {
                        "arm": "A" if kind == "flat" else "B",
                        "doc_id": doc["doc_id"],
                        "chunk_id": c["id"],
                        "kind": c["kind"],
                        "tokens": count,
                        "nodes": len(ns),
                        "pages": sorted({s["page_number"] for s in c["sources"]}),
                        "page_count": len({s["page_index"] for s in c["sources"]}),
                        "headings": c.get("headings", []),
                        "parents": len(parents),
                        "sections": len(sections),
                        "contains_split_multipage_node": split_multipage,
                        "crosses_asset": crosses_asset,
                        "overflow": count > config.chunking.max_tokens,
                    }
                )
        doc_rows.append(
            {
                "doc_id": doc["doc_id"],
                "pages": doc["pages"],
                "normalization_content_equal": invariant,
                "normalization_read_order_equal": order_preserved,
                "docling_nodes": len(nodes["structured"]),
                "multipage_nodes": len(multipage_nodes),
                "charspan_warnings": charspans_invalid,
                "body_nodes": sum(
                    n["kind"] in BODY_KINDS and n["text"].strip() != "" for n in nodes["structured"]
                ),
                "raw_parse_id": raw["id"],
                "normalized_parse_id": parsed["structured"]["id"],
            }
        )
        details[dv] = {"doc": doc, "nodes": nodes, "chunks": {k: by_doc[k][dv] for k in indices}}

    integrity = store.db.execute("PRAGMA integrity_check").fetchone()[0]
    foreign = len(store.db.execute("PRAGMA foreign_key_check").fetchall())
    if integrity != "ok" or foreign:
        issue("sqlite_integrity", integrity=integrity, foreign_key_errors=foreign)
    for kind, index in indices.items():
        fts = store.fts_name(index["id"])
        indexed = list(store.db.execute(f"SELECT id,text FROM {fts}"))
        if len(indexed) != len(chunks[kind]) or dict(indexed) != {c["id"]: c["text"] for c in chunks[kind]}:
            issue("fts_chunk_text_consistency", index=kind)
    mapping_path = Path(manifest["prepared_dir"]) / "mapping.json"
    mapping = read_json(mapping_path)["page_map"] if mapping_path.exists() else {}
    expected_pages = {(d["doc_id"], p) for d in manifest["documents"] for p in range(d["pages"])}
    actual_pages = {(p["doc_id"], p["page_index"]) for p in mapping.values()}
    if mapping and (actual_pages != expected_pages or len(mapping) != len(expected_pages)):
        issue("corpus_page_mapping")
    doc_versions = {d["doc_id"]: d["id"] for d in manifest["documents"]}
    if not mapping and not manifest.get("synthetic"):
        issue("missing_corpus_page_mapping")
    for corpus_id, page in mapping.items():
        if (
            page.get("doc_version") != doc_versions.get(page["doc_id"])
            or page["page_number"] != page["page_index"] + 1
            or str(page["corpus_id"]) != str(corpus_id)
        ):
            issue("corpus_page_binding", corpus_id=corpus_id)
    from .audit_reads import check_reads

    read_checks = check_reads(config, store, indices, details, issues)
    for check in read_checks:
        if not check["passed"]:
            issue("source_readback", details=check)
    per_kind = {}
    for kind in ("A", "B"):
        selected = [c for c in chunk_rows if c["arm"] == kind]
        per_kind[kind] = {
            "chunks": len(selected),
            "tokens": distribution([c["tokens"] for c in selected]),
            "under_100_tokens": sum(c["tokens"] < 100 for c in selected),
            "over_token_target": sum(c["tokens"] > config.chunking.max_tokens for c in selected),
            "token_target": config.chunking.max_tokens,
            "multi_page_chunks": sum(c["page_count"] > 1 for c in selected),
            "split_multipage_node_chunks": sum(c["contains_split_multipage_node"] for c in selected),
            "body_asset_boundary_flags": sum(c["crosses_asset"] for c in selected),
        }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["id"],
        "revision": manifest["revision"],
        "index_ids": {k: v["id"] for k, v in indices.items()},
        "scope": "Source-only, read-only audit. Native-text overlap is a diagnostic, not accuracy or OCR ground truth.",
        "tokenizer": config.tokenizer.model_dump(),
        "documents": doc_rows,
        "indices": per_kind,
        "sqlite_integrity": integrity,
        "foreign_key_errors": foreign,
        "corpus_pages_checked": len(mapping),
        "issues_by_type": dict(Counter(i["check"] for i in issues)),
        "errors": sum(i["severity"] == "error" for i in issues),
        "warnings": sum(i["severity"] == "warning" for i in issues),
        "native_token_recovery": distribution(
            [r["native_token_recovery"] for r in page_rows if r["native_token_recovery"] is not None]
        ),
        "pages_below_90pct_native_token_recovery": sum(
            r["native_token_recovery"] < 0.9 for r in page_rows if r["native_token_recovery"] is not None
        ),
        "unindexed_nonempty_nodes": len(omitted),
        "unindexed_by_reason": dict(Counter(r["reason"] for r in omitted)),
        "tables": {
            "count": len(table_rows),
            "empty": sum(r["empty"] for r in table_rows),
            "without_header_cells": sum(r["header_cells"] == 0 for r in table_rows),
            "chunks_missing_declared_header_terms": sum(
                r["chunks_missing_declared_header_terms"] for r in table_rows
            ),
            "numeric_token_loss_flags": sum(
                r["missing_cell_numeric_tokens_in_chunks"] > 0 for r in table_rows
            ),
        },
        "fragment_page_audit": {
            "checked": len(fragment_rows),
            "aligned": sum(r["status"] == "aligned" for r in fragment_rows),
            "overattributed": sum(bool(r.get("extra_node_pages")) for r in fragment_rows),
        },
        "source_readback": {"cases": len(read_checks), "passed": sum(r["passed"] for r in read_checks)},
        "api_calls_added": store.db.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] - api_before,
    }
    write_csv(output / "pages.csv", page_rows)
    write_csv(output / "chunks.csv", chunk_rows)
    write_csv(output / "tables.csv", table_rows)
    write_json(output / "issues.json", issues)
    write_json(output / "unindexed-nodes.json", omitted)
    write_json(output / "fragment-pages.json", fragment_rows)
    write_json(output / "source-readback.json", read_checks)
    write_json(output / "summary.json", report)
    if gallery:
        from .audit_gallery import build_gallery

        samples = build_gallery(config, store, output, details, page_rows, chunk_rows, table_rows)
    report["visual_samples"] = len(samples)
    write_json(output / "summary.json", report)
    return report
