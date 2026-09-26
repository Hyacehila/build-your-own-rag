"""Bounded official MinerU trial; no model SDK, no automatic paid fallback/retry."""

import os
from urllib.parse import urlparse

import httpx
import pymupdf

from .common import RagError, digest, read_json, sha256, write_json
from .dataset import verify_document

BASE_URL = "https://mineru.net/api/v4"
MAX_TRIAL_PAGES = 8


def prepare_trial(store, selections, output):
    """Each uploaded page keeps its original dimensions and a reversible mapping."""
    if not 1 <= len(selections) <= MAX_TRIAL_PAGES or len(set(selections)) != len(selections):
        raise RagError("MinerU diagnostic trial requires 1-8 distinct physical pages.")
    if (output / "manifest.json").exists():
        saved = read_json(output / "manifest.json")
        if [(p["doc_id"], p["page_number"]) for p in saved["files"]] != list(selections):
            raise RagError("Trial directory belongs to different page selections.")
        return saved
    docs = {d["doc_id"]: d for d in store.meta("dataset")["documents"]}
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, number in selections:
        doc = docs.get(name)
        if not doc or not 1 <= number <= doc["pages"]:
            raise RagError("MinerU trial page is outside the pinned source PDF.")
        verify_document(doc)
        filename = f"{name}-p{number}.pdf"
        dest = output / filename
        with pymupdf.open(doc["pdf_path"]) as pdf, pymupdf.open() as part:
            part.insert_pdf(pdf, from_page=number - 1, to_page=number - 1)
            part.save(dest, no_new_id=True)
            size = list(pdf[number - 1].rect)
        files.append(
            {
                "doc_id": name,
                "doc_version": doc["id"],
                "original_sha256": doc["sha256"],
                "page_number": number,
                "source_page_index": number - 1,
                "uploaded_page_index": 0,
                "page_size": [size[2], size[3]],
                "filename": filename,
                "sha256": sha256(dest),
                "data_id": digest([doc["id"], number])[:32],
            }
        )
    manifest = {
        "contract": "mineru-eight-page-diagnostic-v1",
        "files": files,
        "model_version": "vlm",
        "language": "en",
        "enable_table": True,
        "enable_formula": False,
        "page_limit": MAX_TRIAL_PAGES,
        "official_endpoint": BASE_URL,
        "free_quota_only": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _request(client, method, endpoint, **kwargs):
    key = os.environ.get("MINERU_API_KEY")
    if not key:
        raise RagError("Configure MINERU_API_KEY in the ignored project .env.")
    try:
        response = client.request(
            method, BASE_URL + endpoint, headers={"Authorization": f"Bearer {key}"}, **kwargs
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as e:
        # Do not echo request headers, signed URLs, or provider response bodies.
        raise RagError(f"MinerU request failed ({type(e).__name__}); no automatic retry.") from e
    if payload.get("code") != 0:
        code = str(payload.get("code"))[:40]
        raise RagError(f"MinerU rejected the request (code {code}); no paid fallback or retry.")
    return payload["data"]


def submit_trial(bundle, client=None):
    manifest = read_json(bundle / "manifest.json")
    files = manifest["files"]
    if (
        manifest.get("contract") != "mineru-eight-page-diagnostic-v1"
        or not 1 <= len(files) <= MAX_TRIAL_PAGES
    ):
        raise RagError("Invalid or over-budget MinerU trial manifest.")
    if (bundle / "job.json").exists():
        raise RagError(
            "This trial already has a submission record; use mineru-fetch, never resubmit blindly."
        )
    for entry in files:
        path = (bundle / entry["filename"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != entry["sha256"]:
            raise RagError("Trial PDF checksum/path mismatch.")
        with pymupdf.open(path) as pdf:
            if len(pdf) != 1:
                raise RagError("Every diagnostic upload must contain exactly one page.")
    if not os.environ.get("MINERU_API_KEY"):
        raise RagError("Configure MINERU_API_KEY in the ignored project .env.")
    owned = client is None
    client = client or httpx.Client(timeout=60)
    job = {"state": "submission_reserved", "reserved_pages": len(files), "uploaded": [], "post_attempts": 1}
    write_json(bundle / "job.json", job)
    try:
        payload = {k: manifest[k] for k in ("model_version", "language", "enable_table", "enable_formula")}
        payload["files"] = [{"name": x["filename"], "data_id": x["data_id"], "is_ocr": False} for x in files]
        data = _request(client, "POST", "/file-urls/batch", json=payload)
        job.update(batch_id=data["batch_id"], state="uploading")
        write_json(bundle / "job.json", job)
        urls = data["file_urls"]
        if len(urls) != len(files):
            raise RagError("MinerU returned an unexpected number of upload URLs.")
        for entry, url in zip(files, urls):
            if urlparse(url).scheme != "https":
                raise RagError("MinerU upload requires HTTPS.")
            with (bundle / entry["filename"]).open("rb") as handle:
                response = client.put(url, content=handle)  # no Authorization header on the upload host
                response.raise_for_status()
            job["uploaded"].append(entry["data_id"])
            write_json(bundle / "job.json", job)
        job["state"] = "submitted"
        write_json(bundle / "job.json", job)
        return job
    except (RagError, httpx.HTTPError, KeyError) as e:
        job["state"] = "stopped"
        job["error_type"] = type(e).__name__
        if isinstance(e, RagError):
            job["error"] = str(e)
        write_json(bundle / "job.json", job)
        if isinstance(e, RagError):
            raise
        raise RagError("MinerU upload stopped; submission record retained, no automatic retry.") from e
    finally:
        if owned:
            client.close()


def fetch_trial(bundle, client=None):
    manifest = read_json(bundle / "manifest.json")
    job = read_json(bundle / "job.json")
    if not job.get("batch_id"):
        raise RagError("No confirmed MinerU batch ID; inspect the stopped submission.")
    owned = client is None
    client = client or httpx.Client(timeout=60)
    try:
        data = _request(client, "GET", "/extract-results/batch/" + job["batch_id"])
        by_name = {x["filename"]: x for x in manifest["files"]}
        results = []
        for item in data["extract_result"]:
            entry = by_name.get(item["file_name"])
            if not entry or item.get("data_id", entry["data_id"]) != entry["data_id"]:
                raise RagError("MinerU result does not match a submitted source.")
            row = {"data_id": entry["data_id"], "file_name": entry["filename"], "state": item["state"]}
            if item["state"] == "done":
                dest = bundle / "results" / (entry["data_id"] + ".zip")
                if not dest.exists():
                    url = item["full_zip_url"]
                    if urlparse(url).scheme != "https":
                        raise RagError("MinerU result requires HTTPS.")
                    dest.parent.mkdir(exist_ok=True)
                    tmp = dest.with_suffix(".part")
                    size = 0
                    with client.stream("GET", url, follow_redirects=True) as response, tmp.open("wb") as f:
                        response.raise_for_status()
                        for part in response.iter_bytes():
                            size += len(part)
                            if size > 100_000_000:
                                raise RagError("Single-page result exceeds the diagnostic size limit.")
                            f.write(part)
                    tmp.replace(dest)
                row.update(zip_file=str(dest), sha256=sha256(dest))
            elif item["state"] == "failed":
                row["error"] = "Provider reported parsing failure; not retried."
            results.append(row)
        report = {
            "batch_id": job["batch_id"],
            "results": results,
            "submitted_pages": job["reserved_pages"],
            "complete": len(results) == len(manifest["files"])
            and all(x["state"] in {"done", "failed"} for x in results),
        }
        write_json(bundle / "results.json", report)
        return report
    except httpx.HTTPError as e:
        raise RagError(
            f"MinerU fetch failed ({type(e).__name__}); safe to fetch the same batch again."
        ) from e
    finally:
        if owned:
            client.close()
