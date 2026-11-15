from __future__ import annotations

from pathlib import Path
from time import perf_counter

from financial_rag.common import RagError, digest, read_json, sha256, versions, write_json
from financial_rag.config import Config
from financial_rag.dataset import dataset
from financial_rag.storage import Store


def parse_signature(config: Config, kind: str) -> dict:
    result = {
        "parser": kind,
        "contract": "native-sort-v1" if kind == "flat" else "docling-content-layers-v2",
        "options": config.parsing.semantic_options() if kind == "structured" else {"sort": True},
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
                {"path": p.relative_to(root).as_posix()}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ]
            if root and root.exists()
            else None
        )
    return result


def signature_matches(saved: dict, requested: dict) -> bool:
    import copy
    saved, requested = copy.deepcopy(saved), copy.deepcopy(requested)
    # Historical checksum fields are provenance only, never an acceptance gate.
    for value in (saved, requested):
        value.pop("weights_manifest_sha256", None)
        for weight in value.get("ocr_weights") or []:
            weight.pop("sha256", None)
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


def _require_body_layers(document):
    legacy_root = document.__dict__.get("furniture")
    if legacy_root is not None and legacy_root.children:
        raise RagError("Legacy Docling furniture root is unsupported; reparse the original PDF.")


def docling_nodes(dl_doc, doc: dict, parse_id: str) -> list[dict]:
    from docling_core.types.doc import ContentLayer, PictureItem, TableItem

    from financial_rag.document.facts import relations

    nodes, items = [], {}
    _require_body_layers(dl_doc)
    for item, _ in dl_doc.iterate_items(
        root=dl_doc.body, with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
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
                from financial_rag.document.parsing_guards import GuardedReadingOrderModel

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
        from financial_rag.document.models import verify_models

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
        raise RagError("This workspace contains a legacy fact release; use a new workspace.")
    manifest = dataset(store, verify_pdfs=True)
    signature = parse_signature(config, kind)
    converter = None
    result = {"kind": kind, "completed": [], "cached": [], "failed": []}
    for doc in manifest["documents"]:
        key = f"parse:{kind}:{doc['id']}"
        old = store.meta(key)
        if old:
            artifact = store.get("parses", old)
            if kind == "structured" and artifact.get("normalization"):
                raw_id = store.meta(f"rawparse:structured:{doc['id']}")
                if raw_id:
                    try:
                        raw = store.get("parses", raw_id)
                    except RagError:
                        raw = None
                    if (raw and raw.get("status") == "success" and not raw.get("fixture")
                            and signature_matches(raw.get("semantic_signature", raw.get("signature", {})), signature)):
                        try:
                            load_docling(raw)
                        except (RagError, ValueError, OSError, KeyError):
                            pass
                        else:
                            store.set_meta(key, raw_id)
                            result["cached"].append(doc["doc_id"])
                            continue
            if (
                signature_matches(
                    artifact.get("semantic_signature", artifact.get("signature", {})), signature
                )
                and artifact["status"] == "success"
                and not artifact.get("fixture")
                and not artifact.get("external_document")
                and not artifact.get("normalization")
            ):
                snapshot = Path(artifact["snapshot"])
                readable = snapshot.is_file()
                if readable:
                    try:
                        load_docling(artifact) if kind == "structured" else read_json(snapshot)
                    except (RagError, ValueError, OSError, KeyError):
                        readable = False
                if readable:
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
                store.set_meta(f"rawparse:structured:{doc['id']}", pid)
        result["completed" if parsed["status"] == "success" else "failed"].append(doc["doc_id"])
    write_json(config.work_dir / f"parse-{kind}.json", result)
    return result


def active_parses(config: Config, store: Store, kind: str) -> list[dict]:
    manifest = dataset(store)
    artifacts = []
    signature = parse_signature(config, kind)
    release = store.meta("fact_release") if kind == "structured" else None
    if release:
        raise RagError("This workspace contains a legacy fact release; use a new workspace.")
    for doc in manifest["documents"]:
        pid = store.meta(f"parse:{kind}:{doc['id']}")
        if not pid:
            raise RagError(f"Run parse --parser {kind} for {doc['filename']} first.")
        parsed = store.get("parses", pid)
        if parsed["doc_version"] != doc["id"]:
            raise RagError("Active parse belongs to another document.")
        elif not parsed.get("external_document") and not signature_matches(
            parsed.get("semantic_signature", parsed.get("signature", {})), signature
        ):
            raise RagError(
                f"Parser configuration changed; run parse --parser {kind} to rebuild the snapshot."
            )
        if parsed["status"] != "success":
            raise RagError(f"Incomplete {kind} parse for {doc['filename']}; inspect and fix before chunking.")
        snapshot = Path(parsed["snapshot"])
        if not snapshot.is_file():
            raise RagError("Parse snapshot changed; rerun the parse stage.")
        artifacts.append(parsed)
    return artifacts


def load_docling(parsed: dict):
    from docling_core.types.doc import DoclingDocument

    document = DoclingDocument.model_validate(read_json(Path(parsed["snapshot"])))
    _require_body_layers(document)
    return document
