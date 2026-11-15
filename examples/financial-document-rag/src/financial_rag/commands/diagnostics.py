"""Dataset diagnostics and source inspection command adapters."""

import base64
from pathlib import Path

from ..common import RagError, write_json


def register(commands):
    p = commands.add_parser("source", help="Inspect a raw physical page and associated parsed nodes")
    p.add_argument("--doc-id", required=True)
    p.add_argument("--page", type=int, required=True, help="Physical display page number (1 based)")
    p.set_defaults(handler=source)
    p = commands.add_parser("doctor", help="Check local configuration, storage and source relations")
    p.set_defaults(handler=doctor)

    p = commands.add_parser("diagnose", help="Inspect local data and source quality")
    diagnostics = p.add_subparsers(dest="diagnostic", required=True)
    p = diagnostics.add_parser(
        "audit-data", help="Read-only source, chunk and provenance audit; no model calls"
    )
    p.add_argument("--output", type=Path, help="Default: <work_dir>/data-quality-audit")
    p.add_argument("--no-gallery", action="store_true", help="Only full-corpus automated checks")
    p.set_defaults(handler=audit, read_only=True)


def audit(args, config, store):
    from ..diagnostics.audit import audit_data

    return audit_data(
        config,
        store,
        (args.output or config.work_dir / "data-quality-audit").resolve(),
        gallery=not args.no_gallery,
    )


def source(args, config, store):
    from financial_rag.document.parsing import page_source
    from financial_rag.query.sources import render_source

    from ..dataset import dataset, verify_document

    manifest = dataset(store)
    doc = next((d for d in manifest["documents"] if d["doc_id"] == args.doc_id), None)
    if not doc:
        raise RagError("Unknown doc ID. Find document IDs in inspection.json or the dataset manifest.")
    verify_document(doc)
    page = page_source(doc, args.page - 1)
    data = render_source(store, page, config.query.image_scale)
    dest = config.work_dir / "inspection" / doc["id"] / f"page-{args.page}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
    nodes = []
    for kind in ("flat", "structured"):
        pid = store.meta(f"parse:{kind}:{doc['id']}")
        if pid:
            nodes.extend(
                n for n in store.nodes(pid) if any(s["page_index"] == args.page - 1 for s in n["sources"])
            )
    write_json(dest.with_suffix(".json"), {"source": page, "nodes": nodes})
    return {"image": str(dest.with_suffix(".png")), "nodes": str(dest.with_suffix(".json")), "source": page}


def doctor(args, config, store):
    from ..validation import doctor

    return doctor(config, store)
