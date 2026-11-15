"""Standalone tools for source-preserving DoclingDocument cleaning."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path

if os.name == "nt":
    # PyTorch's compiled backend otherwise requires a separately installed MSVC toolchain.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pymupdf
from clean_documents.asset_roles import empty_tables_as_pictures
from clean_documents.cleaning_profiles import apply_profile
from clean_documents.mineru_bridge import convert_content_list
from clean_documents.mineru_cloud import fetch_trial
from clean_documents.mineru_layout_bridge import convert_document
from clean_documents.mineru_packets import prepare_document, submit_document
from clean_documents.parsing_quality import inspect_snapshot
from clean_documents.table_repairs import repair_invalid_spans, repair_tables
from docling_core.types.doc import DoclingDocument
from PIL import Image
from pydantic import AnyUrl


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def embed_images(document, directory):
    """Validate and embed local assets so the result survives relocation."""
    root = directory.resolve()
    for item in [*document.pictures, *document.tables, *document.pages.values()]:
        image = getattr(item, "image", None)
        if image is None:
            continue
        uri = image.uri
        if isinstance(uri, AnyUrl):
            if uri.scheme != "data":
                raise ValueError("Images must be relative files or embedded data")
            header, separator, content = str(uri).partition(",")
            if not separator or header.lower() != f"data:{image.mimetype};base64".lower():
                raise ValueError("Invalid embedded image data URI")
            data = base64.b64decode(content, validate=True)
        else:
            relative = Path(uri)
            if relative.is_absolute():
                raise ValueError("Images must use relative paths")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Image must be inside the document directory")
            data = path.read_bytes()
            image.uri = AnyUrl(f"data:{image.mimetype};base64,{base64.b64encode(data).decode('ascii')}")
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.verify()


def fact_content(value):
    if isinstance(value, list):
        return [fact_content(item) for item in value]
    if isinstance(value, dict):
        return {key: fact_content(item) for key, item in value.items() if key not in {"parent", "children"}}
    return value


def validate(document, pdf, *, require_all=True):
    with pymupdf.open(pdf) as source:
        count = len(source)
    pages = set(document.pages)
    if not pages or not pages <= set(range(1, count + 1)) or (require_all and len(pages) != count):
        raise ValueError("Invalid original physical-page map")
    document._validate_rules()
    if not document.validate_tree(document.body):
        raise ValueError("Document body has invalid structure")
    for collection in (document.texts, document.tables, document.pictures):
        for item in collection:
            for provenance in item.prov:
                if not 1 <= provenance.page_no <= count:
                    raise ValueError(f"Source page is out of range: {item.self_ref}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pages", nargs=2, type=int, metavar=("FIRST", "LAST"),
                   help="Parse a sample of inclusive physical pages")
    p.add_argument("--ocr", action="store_true", help="Enable OCR for image-based pages")
    p.add_argument("--no-table-structure", action="store_true", help="Disable table structure recognition")
    p.add_argument("--artifacts-path", type=Path, help="Local Docling model weights")
    p = sub.add_parser("apply-profile")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p = sub.add_parser("diagnose")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("repair-tables")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--ref", action="append", required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p = sub.add_parser("mineru-layout")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--doc-id", required=True)
    p.add_argument("--boundaries", type=int, nargs="+", default=[1])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("mineru-prepare")
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--doc-id", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--max-pages", type=int, required=True)
    p.add_argument("--model", default="vlm")
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--page", action="append", type=int)
    p.add_argument("--exclude", action="append", type=int, default=[])
    p.add_argument("--output", type=Path, required=True)
    for name in ("mineru-submit", "mineru-fetch"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
    p = sub.add_parser("check")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p = sub.add_parser("hierarchize")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("review-packet")
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--entry", nargs=4, action="append", required=True,
                   metavar=("DOC_ID", "PDF", "DOCUMENT", "REVIEW"))
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("extract-sqlite")
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--query", required=True, help="Read-only query returning exactly one value")
    p.add_argument("--params", default="[]", help="JSON array of query parameters")
    p.add_argument("--zlib", action="store_true", help="Decompress the selected value with zlib")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("mineru-convert")
    p.add_argument("--content-list", type=Path, required=True)
    p.add_argument("--page-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "parse":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline = PdfPipelineOptions(do_ocr=args.ocr, do_table_structure=not args.no_table_structure)
        if args.artifacts_path:
            pipeline.artifacts_path = args.artifacts_path
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)})
        options = {"page_range": tuple(args.pages)} if args.pages else {}
        result = converter.convert(args.pdf, **options)
        if result.status.value != "success":
            raise ValueError(f"Parsing incomplete: {result.status.value}")
        document = result.document
        validate(document, args.pdf, require_all=not args.pages)
        write(args.output, document.model_dump(mode="json"))
    elif args.action == "extract-sqlite":
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        params = json.loads(args.params)
        if not isinstance(params, list):
            raise ValueError("Query parameters must be a JSON array")
        database = args.database.resolve(strict=True)
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            cursor = connection.execute(args.query, params)
            if cursor.description is None or len(cursor.description) != 1:
                raise ValueError("Query must return exactly one column")
            rows = cursor.fetchmany(2)
            if len(rows) != 1 or not isinstance(rows[0][0], (str, bytes)):
                raise ValueError("Query must return exactly one text or binary value")
            data = rows[0][0].encode("utf-8") if isinstance(rows[0][0], str) else rows[0][0]
        if args.zlib:
            data = zlib.decompress(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(data)
    elif args.action == "manifest":
        entries = []
        ids = set()
        for doc_id, pdf, document_path, review_path in args.entry:
            if not doc_id or doc_id in ids:
                raise ValueError("Document IDs must be nonempty and unique")
            ids.add(doc_id)
            pdf, document_path, review_path = (Path(p).resolve(strict=True) for p in (pdf, document_path, review_path))
            document = DoclingDocument.model_validate(read(document_path))
            validate(document, pdf)
            embed_images(document, document_path.parent)
            review = read(review_path)
            outline = [{"ref": item.self_ref, "text": item.text, "level": getattr(item, "level", None),
                        "parent": item.parent.cref if item.parent else None,
                        "pages": sorted({p.page_no for p in item.prov})}
                       for item in document.texts if item.label.value in {"title", "section_header"}]
            headings = {item["ref"] for item in outline}
            reviewed = review.get("reviewed_heading_refs")
            pages_viewed = review.get("representative_pages_viewed")
            if (review.get("doc_id") != doc_id or review.get("final_tree_reviewed") is not True
                    or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                    or not isinstance(reviewed, list) or len(reviewed) != len(headings)
                    or set(reviewed) != headings or review.get("reviewed_headings") != outline
                    or review.get("confirmed_wrong_parent_remaining") != []
                    or not isinstance(pages_viewed, list) or not pages_viewed
                    or any(type(page) is not int or page not in document.pages for page in pages_viewed)
                    or (not headings and (not isinstance(review.get("no_headings_reason"), str)
                                          or not review["no_headings_reason"].strip()))):
                raise ValueError(f"Incomplete final outline review: {doc_id}")
            base = args.output.resolve().parent
            entries.append({"doc_id": doc_id, "pdf": os.path.relpath(pdf, base),
                            "document": os.path.relpath(document_path, base),
                            "review": os.path.relpath(review_path, base)})
        write(args.output, {"documents": entries})
    elif args.action == "mineru-prepare":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.doc_id):
            raise ValueError("Document ID must contain only letters, numbers, underscores or hyphens")
        with pymupdf.open(args.pdf) as pdf:
            pages = len(pdf)
        doc = {"id": args.doc_id, "doc_id": args.doc_id, "pdf_path": str(args.pdf.resolve()), "pages": pages}
        annotation = {"doc_id": args.doc_id, "pages": [{"page_number": p} for p in args.exclude]}
        prepare_document(doc, annotation, args.output, selected_pages=args.page, language=args.language,
                         model=args.model, is_ocr=args.ocr, max_pages=args.max_pages)
    elif args.action == "mineru-submit":
        submit_document(args.bundle)
    elif args.action == "mineru-fetch":
        fetch_trial(args.bundle)
    elif args.action == "mineru-layout":
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        convert_document(args.bundle, {"doc_id": args.doc_id, "pdf_path": str(args.pdf)},
                         args.output, args.boundaries)
    elif args.action == "mineru-convert":
        document, report = convert_content_list(read(args.content_list), read(args.page_map))
        write(args.output, document.model_dump(mode="json"))
        write(args.report, report)
    else:
        document = DoclingDocument.model_validate(read(args.document))
        validate(document, args.pdf, require_all=args.action != "diagnose")
        embed_images(document, args.document.parent)
        if args.action == "hierarchize":
            before = fact_content(document.model_dump(mode="json"))
            candidate = document.model_copy(deep=True)
            candidate._hierarchize()
            if fact_content(candidate.model_dump(mode="json")) != before:
                raise ValueError("Hierarchy reconstruction changed document content or provenance")
            validate(candidate, args.pdf)
            write(args.output, candidate.model_dump(mode="json"))
        elif args.action == "apply-profile":
            profile = read(args.profile)
            cleaned, decisions = apply_profile(document, profile)
            mapping = decisions["exclusions"]["old_to_new_refs"]
            refs = [mapping[ref] for ref in profile.get("empty_tables_as_pictures", [])]
            cleaned, asset_mapping, asset_decisions = empty_tables_as_pictures(cleaned, refs)
            decisions["asset_ref_mapping"] = asset_mapping
            decisions["asset_reclassifications"] = asset_decisions
            validate(cleaned, args.pdf)
            write(args.output, cleaned.model_dump(mode="json"))
            write(args.decisions, decisions)
        elif args.action == "review-packet":
            headings = [{"ref": item.self_ref, "text": item.text, "level": getattr(item, "level", None),
                         "parent": item.parent.cref if item.parent else None,
                         "pages": sorted({p.page_no for p in item.prov})}
                        for item in document.texts if item.label.value in {"title", "section_header"}]
            write(args.output, {"headings": headings, "review_complete": False})
        elif args.action == "repair-tables":
            candidate = DoclingDocument.model_validate(read(args.candidate))
            cleaned, decisions = repair_tables(document, candidate, set(args.ref), args.pdf)
            cleaned, spans = repair_invalid_spans(cleaned, args.pdf)
            validate(cleaned, args.pdf)
            write(args.output, cleaned.model_dump(mode="json"))
            write(args.report, {"table_repairs": decisions, "span_repairs": spans,
                                "requires_source_review": True})
        elif args.action == "diagnose":
            write(args.output, inspect_snapshot(document.model_dump(mode="json"), args.pdf))
        else:
            print(json.dumps({"pages": len(document.pages), "status": "valid"}))


if __name__ == "__main__":
    main()
