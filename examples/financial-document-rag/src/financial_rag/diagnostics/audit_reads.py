"""Exercise real saved source reads without a model client or database writes."""

from __future__ import annotations

import base64
import hashlib

import pymupdf

from ..sources import Reader, validate_citations


def check_reads(config, store, indices, details, issues):
    cases = []
    for entry in details.values():
        for kind in ("flat", "structured"):
            chunks = entry["chunks"][kind]
            body = [c for c in chunks if c["kind"] == "text"]
            chosen = min(body or chunks, key=lambda c: (-len(c["node_ids"]), c["id"]))
            cases.append(
                (
                    kind,
                    "document_sample",
                    {"chunk_id": chosen["id"], "mode": "window" if kind == "structured" else "node"},
                )
            )
        for n in entry["nodes"]["structured"]:
            if n["kind"] == "table" and not n["text"].strip():
                cases.append(("structured", "empty_table", {"node_id": n["id"]}))
    for i in issues:
        if i["check"] == "text_charspan_bounds":
            cases.append(("structured", "charspan_warning", {"node_id": i["node_id"]}))
        elif i["check"] == "fragment_page_overattribution":
            cases.append(("structured", "fragment_page_warning", {"chunk_id": i["chunk_id"]}))
    output = []
    for kind, reason, args in cases:
        reader = Reader(config, store, indices[kind])
        result = reader.read(**args)
        errors = []
        if "error" in result:
            errors.append(result["error"])
        for evidence in reader.evidence.values():
            source = evidence["sources"][0]
            doc = store.get("documents", source["doc_version"])
            if evidence["type"] == "text":
                if evidence["kind"] == "page":
                    with pymupdf.open(doc["pdf_path"]) as pdf:
                        original = pdf[source["page_index"]].get_text("text", sort=True)
                else:
                    original = store.get("nodes", evidence["source_key"])["text"]
                if evidence["text"] != original[evidence["offset"] : evidence["end_offset"]]:
                    errors.append("returned text differs from saved original interval")
            else:
                with pymupdf.open(doc["pdf_path"]) as pdf:
                    expected = (
                        pdf[source["page_index"]]
                        .get_pixmap(
                            matrix=pymupdf.Matrix(config.query.image_scale, config.query.image_scale),
                            alpha=False,
                        )
                        .tobytes("png")
                    )
                actual = base64.b64decode(evidence["data_url"].split(",", 1)[1])
                if hashlib.sha256(expected).digest() != hashlib.sha256(actual).digest():
                    errors.append("returned image differs from rendering original physical page")
        citations = validate_citations(list(reader.evidence), reader.public_evidence())
        if not reader.evidence or citations["invalid"]:
            errors.append("missing or invalid citation bindings")
        if reader.text_used > config.query.text_tokens or len(reader.images) > config.query.page_images:
            errors.append("read budget exceeded")
        usage = (reader.text_used, len(reader.images), len(reader.evidence))
        reader.read(**args)
        if usage != (reader.text_used, len(reader.images), len(reader.evidence)):
            errors.append("repeated read was not deduplicated")
        evidence = reader.public_evidence()
        output.append(
            {
                "index": kind,
                "reason": reason,
                "arguments": args,
                "errors": errors,
                "passed": not errors,
                "text_tokens": reader.text_used,
                "page_images": len(reader.images),
                "evidence_ids": list(reader.evidence),
                "citation_validity": citations["validity"],
                "source_pages": sorted(
                    {(s["doc_id"], s["page_number"]) for e in evidence for s in e["sources"]}
                ),
                "empty_text_image_fallback": reason == "empty_table"
                and not any(e["type"] == "text" for e in evidence)
                and bool(reader.images),
            }
        )
    return output
