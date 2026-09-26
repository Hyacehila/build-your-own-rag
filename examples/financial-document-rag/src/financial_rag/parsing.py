from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .common import RagError, digest, read_json, sha256, versions, write_json
from .config import Config
from .dataset import dataset
from .storage import Store


def parse_signature(config: Config, kind: str) -> dict:
    weights = Path(config.parsing.artifacts_path) / "manifest.json" if config.parsing.artifacts_path else None
    result = {
        "parser": kind,
        "contract": "native-sort-v1" if kind == "flat" else "docling-standard-pdf-v1",
        "options": config.parsing.semantic_options() if kind == "structured" else {"sort": True},
        "weights_manifest_sha256": sha256(weights)
        if kind == "structured" and weights and weights.exists()
        else None,
        "versions": versions("pymupdf")
        if kind == "flat"
        else versions("docling-slim", "docling-core", "docling-parse", "docling-ibm-models"),
    }
    if kind == "structured" and config.parsing.ocr and config.parsing.ocr_engine == "rapidocr_torch":
        from importlib.metadata import PackageNotFoundError

        try:
            result["ocr_runtime"] = versions("rapidocr", "torch")
        except PackageNotFoundError as e:
            raise RagError("RapidOCR is optional; run with uv run --extra ocr and local OCR weights.") from e
        root = Path(config.parsing.artifacts_path) / "RapidOcr" if config.parsing.artifacts_path else None
        result["ocr_weights"] = (
            [
                {"path": p.relative_to(root).as_posix(), "sha256": sha256(p)}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ]
            if root and root.exists()
            else None
        )
    return result


def signature_matches(saved: dict, requested: dict) -> bool:
    saved, requested = dict(saved), dict(requested)
    # Restoring verified snapshots does not require installing neural model weights.
    # If weights ARE present, a changed manifest invalidates the cache.
    if requested.get("weights_manifest_sha256") is None:
        saved.pop("weights_manifest_sha256", None)
        requested.pop("weights_manifest_sha256", None)
    return saved == requested


def page_source(doc: dict, page_index: int, ref: str | None = None, bbox=None) -> dict:
    if not 0 <= page_index < doc["pages"]:
        raise RagError(f"Source page out of range: {doc['doc_id']} / {page_index}")
    return {
        "doc_id": doc["doc_id"],
        "doc_version": doc["id"],
        "page_index": page_index,
        "page_number": page_index + 1,
        "node_ref": ref,
        "bbox": bbox,
        "coordinate_system": "PDF points, top-left" if bbox else None,
    }


def flat_document(doc: dict, parse_id: str) -> tuple[dict, list[dict]]:
    import pymupdf

    pages, nodes, failed = [], [], []
    with pymupdf.open(doc["pdf_path"]) as pdf:
        for i, page in enumerate(pdf):
            ref = f"#/pages/{i}"
            try:
                text = page.get_text("text", sort=True)
                source = page_source(doc, i, ref, list(page.rect))
                nodes.append(
                    {
                        "id": digest([parse_id, ref]),
                        "parse_id": parse_id,
                        "ref": ref,
                        "kind": "page",
                        "text": text,
                        "sources": [source],
                        "parent_ref": None,
                        "children": [],
                        "headings": [],
                        "related_refs": [],
                    }
                )
                pages.append({"page_index": i, "text": text})
            except Exception as e:
                failed.append({"page_index": i, "error": f"{type(e).__name__}: {e}"})
    return {"pages": pages, "failed_pages": failed}, nodes


def docling_nodes(dl_doc, doc: dict, parse_id: str) -> list[dict]:
    from docling_core.types.doc import ContentLayer, PictureItem, TableItem

    from .facts import relations

    nodes, items = [], {}
    for root in (dl_doc.body, dl_doc.furniture):
        for item, _ in dl_doc.iterate_items(
            root=root, with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
        ):
            items[item.self_ref] = item
    # Persist even non-traversed containers and associated captions/field nodes.
    for value in dl_doc.__dict__.values():
        for item in value if isinstance(value, list) else []:
            if hasattr(item, "self_ref"):
                items.setdefault(item.self_ref, item)
    for item in items.values():
        sources = []
        for prov in getattr(item, "prov", []):
            height = dl_doc.pages[prov.page_no].size.height
            box = prov.bbox.to_top_left_origin(page_height=height)
            sources.append(page_source(doc, prov.page_no - 1, item.self_ref, [box.l, box.t, box.r, box.b]))
        kind = item.label.value if hasattr(item, "label") else "group"
        caption = item.caption_text(dl_doc) if isinstance(item, (TableItem, PictureItem)) else ""
        related = [r.cref for r in getattr(item, "captions", [])] + [
            r.cref for r in getattr(item, "footnotes", [])
        ]
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=dl_doc)
        else:
            text = getattr(item, "text", "") or caption
        nodes.append(
            {
                "id": digest([parse_id, item.self_ref]),
                "parse_id": parse_id,
                "ref": item.self_ref,
                "kind": kind,
                "text": text,
                "caption": caption,
                "sources": sources,
                "headings": [],
                "parent_ref": item.parent.cref if item.parent else None,
                "children": [r.cref for r in item.children],
                "related_refs": related,
                "raw": item.model_dump(mode="json", exclude_none=True),
            }
        )
    refs = {n["ref"]: n for n in nodes}
    for node in nodes:
        for ref in node["related_refs"]:
            if ref not in refs:
                raise RagError(f"Unresolved caption/footnote reference: {ref}")
        if node["kind"] in ("table", "picture"):
            # Caption evidence is part of the same retrievable asset, with its own provenance.
            for ref in node["related_refs"]:
                refs[ref]["related_refs"].append(node["ref"])
    return relations(nodes)


def _converter(config: Config):
    import threading

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    class ProgressPdfPipeline(StandardPdfPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if config.parsing.guarded_merges:
                from .parsing_guards import GuardedReadingOrderModel

                self.reading_order_model = GuardedReadingOrderModel(self.reading_order_model.options)

        def _build_document(self, conv_res):
            stop = threading.Event()

            def progress():
                while not stop.wait(30):
                    pages = list(conv_res.pages)
                    completed = sum(p.assembled is not None for p in pages)
                    print(
                        f"Docling {conv_res.input.file.name}: {completed}/{len(pages)} pages assembled",
                        flush=True,
                    )

            observer = threading.Thread(target=progress, daemon=True)
            observer.start()
            try:
                return super()._build_document(conv_res)
            finally:
                stop.set()
                observer.join(timeout=1)

    options = PdfPipelineOptions(
        do_ocr=config.parsing.ocr,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        generate_parsed_pages=config.parsing.heading_hierarchy,
        force_backend_text=config.parsing.force_backend_text,
    )
    options.heading_hierarchy_options.enabled = config.parsing.heading_hierarchy
    options.accelerator_options = AcceleratorOptions(
        device=config.parsing.device, num_threads=config.parsing.threads
    )
    options.layout_options.engine_options.compile_model = config.parsing.compile_model
    options.table_structure_options.mode = TableFormerMode(config.parsing.table_mode)
    options.table_structure_options.do_cell_matching = config.parsing.table_cell_matching
    if config.parsing.ocr:
        if config.parsing.ocr_engine == "rapidocr_torch":
            from docling.datamodel.pipeline_options import RapidOcrOptions

            options.ocr_options = RapidOcrOptions(lang=["en"], backend="torch", mode=config.parsing.ocr_mode)
        else:
            options.ocr_options = TesseractCliOcrOptions(lang=["eng"], mode=config.parsing.ocr_mode)
    if config.parsing.artifacts_path:
        from .models import verify_models

        verify_models(Path(config.parsing.artifacts_path))
        options.artifacts_path = Path(config.parsing.artifacts_path)
    backend = {}
    if config.parsing.backend == "pdfium":
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        backend["backend"] = PyPdfiumDocumentBackend
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ProgressPdfPipeline, pipeline_options=options, **backend
            )
        },
    )


def parse(config: Config, store: Store, kind: str) -> dict:
    if kind == "structured" and store.meta("fact_release"):
        artifacts = active_parses(config, store, kind)
        return {
            "kind": kind,
            "completed": [],
            "failed": [],
            "cached": [p["doc_version"] for p in artifacts],
            "note": "Reusing accepted facts; rebuild through the cleaning pipeline to change them.",
        }
    manifest = dataset(store, verify_pdfs=True)
    signature = parse_signature(config, kind)
    converter = None
    result = {"kind": kind, "completed": [], "cached": [], "failed": []}
    for doc in manifest["documents"]:
        key = f"parse:{kind}:{doc['id']}"
        old = store.meta(key)
        if old:
            artifact = store.get("parses", old)
            if (
                signature_matches(
                    artifact.get("semantic_signature", artifact.get("signature", {})), signature
                )
                and artifact["status"] == "success"
                and not artifact.get("fixture")
            ):
                snapshot = Path(artifact["snapshot"])
                if snapshot.exists() and sha256(snapshot) == artifact["snapshot_sha256"]:
                    result["cached"].append(doc["doc_id"])
                    continue
        pid = digest({"doc_version": doc["id"], "signature": signature})
        parsed = {
            "id": pid,
            "doc_version": doc["id"],
            "kind": kind,
            "signature": signature,
            "status": "failed",
            "failed_pages": [],
            "errors": [],
        }
        nodes = []
        started = perf_counter()
        print(f"Parsing {kind}: {doc['filename']} ({doc['pages']} pages)", flush=True)
        try:
            if kind == "flat":
                snapshot_data, nodes = flat_document(doc, pid)
                parsed["failed_pages"] = snapshot_data["failed_pages"]
                parsed["status"] = "partial" if parsed["failed_pages"] else "success"
            else:
                if converter is None:
                    converter = _converter(config)
                from docling.datamodel.base_models import InputFormat

                parsed["resolved_options"] = converter.format_to_options[
                    InputFormat.PDF
                ].pipeline_options.model_dump(mode="json")
                converted = converter.convert(Path(doc["pdf_path"]), raises_on_error=False)
                dl_doc = converted.document
                snapshot_data = dl_doc.model_dump(mode="json")
                nodes = docling_nodes(dl_doc, doc, pid)
                parsed["status"] = converted.status.value
                parsed["errors"] = [e.model_dump(mode="json") for e in converted.errors]
                missing = set(range(1, doc["pages"] + 1)) - set(dl_doc.pages)
                parsed["failed_pages"] = [
                    {"page_index": n - 1, "error": "Missing from DoclingDocument"} for n in sorted(missing)
                ]
                for error in parsed["errors"]:
                    if error.get("page_no") is not None:
                        failed_page = int(error["page_no"]) - 1
                        if failed_page not in {p["page_index"] for p in parsed["failed_pages"]}:
                            parsed["failed_pages"].append(
                                {"page_index": failed_page, "error": error.get("error_message", str(error))}
                            )
                # Some conversion errors do not identify a page. Keep them at document scope.
                if parsed["failed_pages"] and parsed["status"] == "success":
                    parsed["status"] = "partial"
            snapshot = config.work_dir / "parses" / pid / "document.json"
            write_json(snapshot, snapshot_data)
            parsed.update(snapshot=str(snapshot), snapshot_sha256=sha256(snapshot))
        except Exception as e:
            parsed["errors"].append({"type": type(e).__name__, "message": str(e)})
        parsed["elapsed_seconds"] = perf_counter() - started
        store.put_parse(parsed, nodes)
        store.set_meta(key, pid)
        if parsed["status"] == "success":
            store.save_snapshot(Path(parsed["snapshot"]), parsed["snapshot_sha256"])
            if kind == "structured":
                from .facts import normalized_parse

                store.set_meta(f"rawparse:structured:{doc['id']}", pid)
                normalized_parse(config, store, parsed, doc)
        result["completed" if parsed["status"] == "success" else "failed"].append(doc["doc_id"])
    write_json(config.work_dir / f"parse-{kind}.json", result)
    return result


def active_parses(config: Config, store: Store, kind: str) -> list[dict]:
    manifest = dataset(store)
    artifacts = []
    signature = parse_signature(config, kind)
    release = store.meta("fact_release") if kind == "structured" else None
    if release:
        from .final_outline import verify_stored_acceptance

        verify_stored_acceptance(store)
        accepted_ids = {row["parse_id"] for row in release["documents"]}
    for doc in manifest["documents"]:
        pid = store.meta(f"parse:{kind}:{doc['id']}")
        if not pid:
            raise RagError(f"Run parse --parser {kind} for {doc['filename']} first.")
        parsed = store.get("parses", pid)
        if release:
            if (
                pid not in accepted_ids
                or parsed["doc_version"] != doc["id"]
                or parsed.get("cleaning_contract") != release["contract"]
            ):
                raise RagError("Active structured parse is not part of the accepted fact release.")
        elif not signature_matches(
            parsed.get("semantic_signature", parsed.get("signature", {})), signature
        ) and not (manifest["synthetic"] and parsed.get("fixture")):
            raise RagError(
                f"Parser contract changed or legacy cache needs migration; run migrate before considering parse --parser {kind}."
            )
        if parsed["status"] != "success":
            raise RagError(f"Incomplete {kind} parse for {doc['filename']}; inspect and fix before chunking.")
        snapshot = Path(parsed["snapshot"])
        if not snapshot.exists() or sha256(snapshot) != parsed["snapshot_sha256"]:
            raise RagError("Parse snapshot changed; rerun the parse stage.")
        artifacts.append(parsed)
    return artifacts


def load_docling(parsed: dict):
    from docling_core.types.doc import DoclingDocument

    return DoclingDocument.model_validate(read_json(Path(parsed["snapshot"])))


def parse_sample(
    config: Config, store: Store, doc_id: str, start: int, end: int, *, profile="configured", refresh=False
) -> dict:
    """Real parser diagnostic on physical pages, never published as a full parse/index."""
    from collections import Counter

    from docling.datamodel.base_models import InputFormat

    from .diagnostics.parsing_quality import QUALITY_CONTRACT, inspect_snapshot

    config = config.model_copy(deep=True)
    profiles = {
        "configured": {},
        "guarded": {"guarded_merges": True},
        "pdfium": {"backend": "pdfium"},
        "no-cell-match": {"table_cell_matching": False},
        "bitmap-ocr": {"ocr": True, "ocr_engine": "rapidocr_torch", "ocr_mode": "default"},
        "full-ocr": {"ocr": True, "ocr_engine": "rapidocr_torch", "ocr_mode": "full_page"},
    }
    if profile not in profiles:
        raise RagError("Unknown diagnostic parsing profile")
    for key, value in profiles[profile].items():
        setattr(config.parsing, key, value)

    manifest = dataset(store, verify_pdfs=True)
    doc = next((d for d in manifest["documents"] if d["doc_id"] == doc_id), None)
    if not doc or not 1 <= start <= end <= doc["pages"]:
        raise RagError("Choose an existing doc ID and an inclusive physical page range within that PDF.")
    if end - start + 1 > 10:
        raise RagError("parse-sample is limited to 10 physical pages; split the diagnostic range.")
    signature = parse_signature(config, "structured")
    pid = digest(["diagnostic_sample_v2", QUALITY_CONTRACT, doc["id"], start, end, signature])
    output = config.work_dir / "samples" / pid
    if not refresh and (output / "report.json").is_file():
        cached = read_json(output / "report.json")
        if (
            not cached["failed"]
            and all(
                (output / name).is_file() and sha256(output / name) == checksum
                for name, checksum in cached.get("files", {}).items()
            )
            and cached.get("files")
        ):
            return {**cached, "cached": True}
    converter = _converter(config)
    converter.format_to_options[InputFormat.PDF].pipeline_options.generate_parsed_pages = True
    started = perf_counter()
    result = converter.convert(Path(doc["pdf_path"]), page_range=(start, end), raises_on_error=False)
    nodes = docling_nodes(result.document, doc, pid)
    expected = set(range(start, end + 1))
    pages = set(result.document.pages)
    report = {
        "diagnostic_only": True,
        "profile": profile,
        "doc_id": doc_id,
        "doc_version": doc["id"],
        "page_range": [start, end],
        "parsed_pages": sorted(pages),
        "elapsed_seconds": perf_counter() - started,
        "status": result.status.value,
        "errors": [e.model_dump(mode="json") for e in result.errors],
        "failed_pages": sorted(expected - pages),
        "node_types": dict(Counter(n["kind"] for n in nodes)),
        "signature": signature,
        "snapshot": str(output / "document.json"),
        "nodes": str(output / "nodes.json"),
    }
    report["failed"] = result.status.value != "success" or pages != expected
    snapshot = result.document.model_dump(mode="json")
    write_json(output / "document.json", snapshot)
    write_json(output / "nodes.json", nodes)
    stages = [
        {
            "page_no": page.page_no,
            "predictions": page.predictions.model_dump(mode="json"),
            "parsed_page": page.parsed_page.model_dump(mode="json") if page.parsed_page else None,
        }
        for page in result.pages
    ]
    write_json(output / "stages.json", stages)
    quality = inspect_snapshot(snapshot, doc["pdf_path"])
    write_json(output / "quality.json", quality)
    report["quality_counts"] = quality["counts"]
    report["needs_quality_review"] = bool(quality["issues"])
    if config.parsing.guarded_merges:
        report["reading_order_changes"] = [
            {
                "rejected_merges": pipeline.reading_order_model.rejected_merges,
                "zones": pipeline.reading_order_model.ro_model.zone_changes,
            }
            for pipeline in converter.initialized_pipelines.values()
        ]
    report["files"] = {
        name: sha256(output / name) for name in ("document.json", "nodes.json", "stages.json", "quality.json")
    }
    write_json(output / "report.json", report)
    return report
