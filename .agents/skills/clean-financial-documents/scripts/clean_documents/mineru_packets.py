"""Explicit one-document MinerU comparison, separate from the eight-page diagnostic."""

import os
from urllib.parse import urlparse

import httpx
import pymupdf

from .common import RagError, digest, read_json, write_json, write_state
from .mineru_cloud import BASE_URL, _request


def prepare_document(doc, annotation, output, core_size=190, overlap=2, selected_pages=None,
                     language="en", model="vlm", is_ocr=False, max_pages=None):
    if type(max_pages) is not int or max_pages < 1:
        raise RagError("Declare a positive maximum number of pages before preparing packets.")
    if not 1 <= core_size <= 196 or not 0 <= overlap <= 2 or core_size + 2 * overlap > 200:
        raise RagError("Document packets must stay below the 200-page API limit.")
    if annotation["doc_id"] != doc["doc_id"]:
        raise RagError("Exclusion annotation belongs to another PDF.")
    excluded = {x["page_number"] for x in annotation["pages"]}
    if not excluded <= set(range(1, doc["pages"] + 1)):
        raise RagError("Excluded page outside source PDF.")
    selected = [n for n in range(1, doc["pages"] + 1) if n not in excluded]
    if selected_pages is not None:
        if not selected_pages or any(type(p) is not int or p not in selected for p in selected_pages):
            raise RagError("Risk selection must contain retained original physical pages.")
        selected = sorted(set(selected_pages))
    if not selected:
        raise RagError("At least one retained page must be selected.")
    count = sum(len(selected[max(0, start - overlap):start + core_size + overlap])
                for start in range(0, len(selected), core_size))
    if count > max_pages:
        raise RagError("Submitted pages including overlap exceed the declared maximum.")
    identity = digest([doc["id"], str(doc["pdf_path"]), doc["pages"], annotation, core_size, overlap,
                       model, language, is_ocr, max_pages, selected])
    if (output / "manifest.json").exists():
        saved = read_json(output / "manifest.json")
        if saved["identity"] != identity:
            raise RagError("Document trial directory has different inputs.")
        return saved
    if output.exists() and any(output.iterdir()):
        raise RagError("Packet directory is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    with pymupdf.open(doc["pdf_path"]) as pdf:
        for part_no, start in enumerate(range(0, len(selected), core_size), 1):
            owner = set(selected[start : start + core_size])
            pages = selected[max(0, start - overlap) : start + core_size + overlap]
            filename = f"{doc['doc_id']}-part-{part_no:02d}.pdf"
            with pymupdf.open() as packet:
                for n in pages:
                    packet.insert_pdf(pdf, from_page=n - 1, to_page=n - 1)
                packet.save(output / filename, no_new_id=True)
            files.append(
                {
                    "filename": filename,
                    "data_id": digest([identity, part_no])[:32],

                    "pages": [
                        {
                            "uploaded_page_index": i,
                            "source_page_index": n - 1,
                            "page_number": n,
                            "page_size": [pdf[n - 1].rect.width, pdf[n - 1].rect.height],
                            "owner": n in owner,
                        }
                        for i, n in enumerate(pages)
                    ],
                }
            )
    manifest = {
        "contract": "mineru-one-document-comparison-v1",
        "identity": identity,
        "doc_id": doc["doc_id"],
        "doc_version": doc["id"],

        "excluded_page_numbers": sorted(excluded),
        "unique_body_pages": len(selected),
        "submitted_pages": count,
        "page_limit": max_pages,
        "files": files,
        "model_version": model,
        "language": language,
        "enable_table": True,
        "enable_formula": True,
        "is_ocr": is_ocr,
        "official_endpoint": BASE_URL,

        "purpose": "User-requested one-document parser comparison; no full RAG experiment",
        "selection": "risk_pages_with_context" if selected_pages is not None else "full_body",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def submit_document(bundle, client=None):
    manifest = read_json(bundle / "manifest.json")
    if manifest["contract"] != "mineru-one-document-comparison-v1":
        raise RagError("Not an authorized document comparison manifest.")
    if (bundle / "job.json").exists():
        raise RagError("Submission already reserved; fetch the recorded batch instead of resubmitting.")
    files = manifest["files"]
    actual_count = 0
    owner_pages = []
    for f in files:
        path = (bundle / f["filename"]).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise RagError("Packet path escapes its bundle.")
        with pymupdf.open(path) as pdf:
            if len(pdf) != len(f["pages"]) or not 1 <= len(pdf) <= 200 or path.stat().st_size > 200_000_000:
                raise RagError("Packet exceeds API limits or page mapping differs.")
            actual_count += len(pdf)
        owner_pages.extend(p["page_number"] for p in f["pages"] if p["owner"])
    if (
        not 1 <= len(files) <= 50
        or not 1 <= actual_count <= manifest["page_limit"]
        or actual_count != manifest["submitted_pages"]
        or len(owner_pages) != len(set(owner_pages))
        or len(owner_pages) != manifest["unique_body_pages"]
        or set(owner_pages) & set(manifest["excluded_page_numbers"])
    ):
        raise RagError("Document page budget or source ownership is invalid.")
    if not os.environ.get("MINERU_API_KEY"):
        raise RagError("Missing MINERU_API_KEY in the ignored .env.")
    owned = client is None
    client = client or httpx.Client(timeout=120)
    job = {"state": "submission_reserved", "reserved_pages": actual_count, "uploaded": [], "post_attempts": 1}
    write_json(bundle / "job.json", job)
    try:
        payload = {k: manifest[k] for k in ("model_version", "language", "enable_table", "enable_formula")}
        payload["files"] = [{"name": f["filename"], "data_id": f["data_id"], "is_ocr": manifest["is_ocr"]} for f in files]
        data = _request(client, "POST", "/file-urls/batch", json=payload)
        job.update(batch_id=data["batch_id"], state="uploading")
        write_state(bundle / "job.json", job)
        if len(data["file_urls"]) != len(files):
            raise RagError("Upload URL count mismatch.")
        for entry, url in zip(files, data["file_urls"]):
            if urlparse(url).scheme != "https":
                raise RagError("HTTPS required for uploads.")
            with (bundle / entry["filename"]).open("rb") as f:
                client.put(url, content=f).raise_for_status()
            job["uploaded"].append(entry["data_id"])
            write_state(bundle / "job.json", job)
        job["state"] = "submitted"
        write_state(bundle / "job.json", job)
        return job
    except Exception as e:
        job.update(state="stopped", error_type=type(e).__name__)
        write_state(bundle / "job.json", job)
        if isinstance(e, RagError):
            raise
        raise RagError("Document upload stopped; reserved batch retained, no automatic retry.") from e
    finally:
        if owned:
            client.close()
