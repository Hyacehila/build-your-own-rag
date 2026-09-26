"""Optional diagnostics, source inspection and bounded trial command adapters."""

import base64
from pathlib import Path

from ..common import RagError, write_json


def register(commands):
    p = commands.add_parser("audit-data", help="Read-only source, chunk and provenance audit; no model calls")
    p.add_argument("--output", type=Path, help="Default: <work_dir>/data-quality-audit")
    p.add_argument("--no-gallery", action="store_true", help="Only full-corpus automated checks")
    p.set_defaults(handler=audit, read_only=True)
    p = commands.add_parser(
        "repair-preview", help="Local deterministic repair candidates; no active DB writes"
    )
    p.add_argument("--doc-id", action="append", required=True)
    p.add_argument("--output", type=Path)
    p.set_defaults(handler=repair_preview, read_only=True)
    p = commands.add_parser("mineru-prepare", help="Prepare at most eight source-mapped single-page PDFs")
    p.add_argument("--page", action="append", required=True, help="doc_id:physical_page_number")
    p.add_argument("--bundle", type=Path, required=True)
    p.set_defaults(handler=mineru, read_only=True)
    for name in ("mineru-submit", "mineru-fetch", "mineru-import"):
        p = commands.add_parser(name, help="Bounded MinerU trial; existing experiment DB stays read-only")
        p.add_argument("--bundle", type=Path, required=True)
        p.set_defaults(handler=mineru, read_only=True)
    p = commands.add_parser("parse-sample", help="Diagnose real Docling parsing on a physical page range")
    p.add_argument("--doc-id", required=True)
    p.add_argument("--start-page", type=int, default=1)
    p.add_argument("--end-page", type=int, default=3)
    p.add_argument(
        "--profile",
        choices=["configured", "guarded", "pdfium", "no-cell-match", "bitmap-ocr", "full-ocr"],
        default="configured",
    )
    p.add_argument(
        "--refresh", action="store_true", help="Rerun this exact local sample instead of its cache"
    )
    p.set_defaults(handler=parse_sample, read_only=True)
    p = commands.add_parser("source", help="Inspect a raw physical page and associated parsed nodes")
    p.add_argument("--doc-id", required=True)
    p.add_argument("--page", type=int, required=True, help="Physical display page number (1 based)")
    p.set_defaults(handler=source)
    p = commands.add_parser("fixture", help="Create a small FICTIONAL PDF for local verification")
    p.add_argument(
        "--structured-snapshot",
        action="store_true",
        help="Also write a known-layout DoclingDocument; does not test the learned PDF parser",
    )
    p.set_defaults(handler=fixture)
    p = commands.add_parser(
        "doctor", help="Check local contracts; --live calls the bounded API protocol check"
    )
    p.add_argument("--live", action="store_true")
    p.set_defaults(handler=doctor)
    p = commands.add_parser("smoke", help="Run only the three fictional questions in A/B/C/D")
    p.add_argument("--run", default="fictional-smoke-v1")
    p.set_defaults(handler=smoke)
    p = commands.add_parser("check-known-pages", help="Check JPMorgan 73/96 and Morgan Stanley 101")
    p.set_defaults(handler=check_known_pages)


def audit(args, config, store):
    from ..diagnostics.audit import audit_data

    return audit_data(
        config,
        store,
        (args.output or config.work_dir / "data-quality-audit").resolve(),
        gallery=not args.no_gallery,
    )


def repair_preview(args, config, store):
    from ..diagnostics.repair_preview import repair_preview

    return repair_preview(
        config, store, args.doc_id, (args.output or config.work_dir / "repair-preview").resolve()
    )


def mineru(args, config, store):
    from ..mineru_bridge import import_trial
    from ..mineru_cloud import fetch_trial, prepare_trial, submit_trial

    bundle = args.bundle.resolve()
    if args.command == "mineru-prepare":
        try:
            selections = [(value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in args.page]
        except (ValueError, IndexError) as e:
            raise RagError("Use --page doc_id:physical_page_number") from e
        return prepare_trial(store, selections, bundle)
    if args.command == "mineru-submit":
        return submit_trial(bundle)
    if args.command == "mineru-fetch":
        return fetch_trial(bundle)
    return import_trial(store, bundle)


def parse_sample(args, config, store):
    from ..parsing import parse_sample

    return parse_sample(
        config,
        store,
        args.doc_id,
        args.start_page,
        args.end_page,
        profile=args.profile,
        refresh=args.refresh,
    )


def source(args, config, store):
    from ..dataset import dataset, verify_document
    from ..parsing import page_source
    from ..sources import render_source

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


def fixture(args, config, store):
    from ..fixture import create_fixture

    return create_fixture(config, store, args.structured_snapshot)


def doctor(args, config, store):
    from ..validation import doctor

    return doctor(config, store, live=args.live)


def smoke(args, config, store):
    from ..validation import smoke

    return smoke(config, store, args.run)


def check_known_pages(args, config, store):
    from ..validation import check_known_pages

    return check_known_pages(config, store)
