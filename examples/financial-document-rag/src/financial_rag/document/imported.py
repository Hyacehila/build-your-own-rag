"""Read reviewed Docling documents from a portable file manifest."""

import base64
import filecmp
import io
import json
from pathlib import Path
from uuid import uuid4

import pymupdf
from docling_core.types.doc import DoclingDocument
from PIL import Image
from pydantic import AnyUrl

from financial_rag.common import RagError, canonical, digest, write_json
from financial_rag.dataset import dataset
from financial_rag.document.facts import relations
from financial_rag.document.parsing import docling_nodes, parse_signature


def embed_images(document, directory):
    """Preserve local image resources when the document moves into the workspace."""
    root = directory.resolve()
    for item in [*document.pictures, *document.tables, *document.pages.values()]:
        image = getattr(item, "image", None)
        if image is None:
            continue
        uri = image.uri
        if isinstance(uri, AnyUrl):
            if uri.scheme != "data":
                raise ValueError("Document image resources must be relative files or embedded data")
            header, separator, content = str(uri).partition(",")
            if not separator or header.lower() != f"data:{image.mimetype};base64".lower():
                raise ValueError("Embedded image must be a base64 data URI of the declared type")
            with Image.open(io.BytesIO(base64.b64decode(content, validate=True))) as decoded:
                decoded.verify()
            continue
        relative = Path(uri)
        if relative.is_absolute():
            raise ValueError("Document image resource must use a relative path")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Document image resource must be inside the delivery directory")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.verify()
        image.uri = AnyUrl(f"data:{image.mimetype};base64,{base64.b64encode(data).decode('ascii')}")


def import_documents(config, store, manifest_path):
    if store.meta("fact_release"):
        raise RagError("This workspace contains a legacy fact release; use a new workspace.")
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = manifest["documents"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("documents must be a list of objects")
        source = dataset(store, verify_pdfs=True, verify_queries=True)
        expected = {doc["doc_id"]: doc for doc in source["documents"]}
        names = [row["doc_id"] for row in rows]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("Manifest document IDs must match the dataset exactly")
        staged = []
        for row in rows:
            doc = expected[row["doc_id"]]
            pdf, document_file, review_file = (path.parent / row[key] for key in ("pdf", "document", "review"))
            pdf, document_file, review_file = (p.resolve(strict=True) for p in (pdf, document_file, review_file))
            review = json.loads(review_file.read_text(encoding="utf-8"))
            with pymupdf.open(pdf) as pdf_doc:
                if len(pdf_doc) != doc["pages"] or pdf.name != doc["filename"]:
                    raise ValueError(f"Source PDF identity or page count differs: {doc['doc_id']}")
            if not filecmp.cmp(pdf, doc["pdf_path"], shallow=False):
                raise ValueError(f"Source PDF differs from the dataset document: {doc['doc_id']}")
            document = DoclingDocument.model_validate_json(document_file.read_text(encoding="utf-8"))
            if set(document.pages) != set(range(1, doc["pages"] + 1)):
                raise ValueError(f"Invalid physical page map: {doc['doc_id']}")
            document._validate_rules()
            if not document.validate_tree(document.body):
                raise ValueError(f"Invalid document tree: {doc['doc_id']}")
            embed_images(document, document_file.parent)
            outline = [{"ref": item.self_ref, "text": item.text, "level": getattr(item, "level", None),
                        "parent": item.parent.cref if item.parent else None,
                        "pages": sorted({p.page_no for p in item.prov})}
                       for item in document.texts if item.label.value in {"title", "section_header"}]
            headings = {item["ref"] for item in outline}
            reviewed = review.get("reviewed_heading_refs")
            pages_viewed = review.get("representative_pages_viewed")
            if (review.get("doc_id") != doc["doc_id"] or review.get("final_tree_reviewed") is not True
                    or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                    or not isinstance(reviewed, list) or len(reviewed) != len(headings)
                    or set(reviewed) != headings or review.get("reviewed_headings") != outline
                    or review.get("confirmed_wrong_parent_remaining") != []
                    or not isinstance(pages_viewed, list) or not pages_viewed
                    or any(type(page) is not int or not 1 <= page <= doc["pages"] for page in pages_viewed)
                    or (not headings and (not isinstance(review.get("no_headings_reason"), str)
                                          or not review["no_headings_reason"].strip()))):
                raise ValueError(f"Incomplete final outline review: {doc['doc_id']}")
            pid = uuid4().hex
            nodes = docling_nodes(document, doc, pid)
            relations(nodes)
            if any(not 0 <= location["page_index"] < doc["pages"] for node in nodes for location in node["sources"]):
                raise ValueError(f"Invalid source page: {doc['doc_id']}")
            staged.append((doc, pid, document.model_dump(mode="json"), nodes, review))
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise RagError(f"Invalid reviewed document manifest: {error}") from error

    updates = []
    for doc, pid, payload, nodes, review in staged:
        snapshot = store.root / "parses" / pid / "document.json"
        write_json(snapshot, payload)
        snapshot_id = digest(payload)  # Content-addressed database key, not a file integrity check.
        parsed = {
            "id": pid, "doc_version": doc["id"], "kind": "structured", "status": "success",
            "signature": parse_signature(config, "structured"), "external_document": True,
            "snapshot": str(snapshot), "snapshot_sha256": snapshot_id,
            "review": review,
        }
        store.save_snapshot(snapshot, snapshot_id)
        store.put_parse(parsed, nodes)
        updates.append((f"parse:structured:{doc['id']}", canonical(pid)))
    with store.db:
        store.db.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", updates
        )
    return {"kind": "structured", "completed": [doc["doc_id"] for doc, *_ in staged], "failed": []}
