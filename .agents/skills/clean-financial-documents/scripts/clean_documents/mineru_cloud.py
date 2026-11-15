import os
from urllib.parse import urlparse

import httpx

from .common import RagError, read_json, write_state

BASE_URL = "https://mineru.net/api/v4"

def _request(client, method, endpoint, **kwargs):
    key = os.environ.get("MINERU_API_KEY")
    if not key:
        raise RagError("Configure MINERU_API_KEY as an environment variable.")
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
                row.update(zip_file=str(dest))
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
        write_state(bundle / "results.json", report)
        return report
    except httpx.HTTPError as e:
        raise RagError(
            f"MinerU fetch failed ({type(e).__name__}); safe to fetch the same batch again."
        ) from e
    finally:
        if owned:
            client.close()
