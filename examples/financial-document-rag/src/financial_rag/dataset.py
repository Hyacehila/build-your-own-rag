from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import RagError, digest, sha256, write_json, write_jsonl
from .config import Config
from .storage import Store

PDF_PAGES = {
    "jpmorgan_chase_2024.pdf": 437,
    "wells_fargo_2024.pdf": 355,
    "citigroup_2024.pdf": 963,
    "goldman_sachs_2024.pdf": 614,
    "bank_of_america_2024.pdf": 305,
    "morgan_stanley_2024.pdf": 268,
}


def hub_local_dir(path: Path) -> Path:
    """HF temporary filenames can exceed Windows' legacy 260-character limit."""
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
    return Path(absolute)


def prepare_annotations(
    queries: list[dict], qrels: list[dict], corpus: list[dict], documents: list[dict], expected_queries: int
) -> tuple[list[dict], list[dict], dict]:
    """Validate complete source mappings before publishing the prepared dataset.

    The source page-number origin is inferred only from *every page* of each PDF,
    never from the (usually sparse) relevant-page annotations.
    """
    english = [q for q in queries if str(q["language"]).strip().lower() == "english"]
    ids = [str(q["query_id"]) for q in english]
    if len(ids) != expected_queries or len(set(ids)) != len(ids):
        raise RagError(
            f"Expected {expected_queries} unique English query IDs, got {len(ids)} / {len(set(ids))}."
        )
    by_doc = defaultdict(list)
    for row in corpus:
        by_doc[str(row["doc_id"])].append(row)
    if set(by_doc) != {d["doc_id"] for d in documents}:
        raise RagError("Corpus and document metadata have different document IDs.")
    mapping, origins = {}, {}
    for doc in documents:
        rows = by_doc[doc["doc_id"]]
        values = [int(r["page_number_in_doc"]) for r in rows]
        count = doc["pages"]
        if len(values) != count or len(set(values)) != count:
            raise RagError(f"Incomplete or duplicated page mapping for {doc['doc_id']}.")
        if set(values) == set(range(count)):
            origin = 0
        elif set(values) == set(range(1, count + 1)):
            origin = 1
        else:
            raise RagError(f"Non-contiguous page mapping for {doc['doc_id']}.")
        origins[doc["doc_id"]] = origin
        for row in rows:
            cid = str(row["corpus_id"])
            if cid in mapping:
                raise RagError(f"Duplicated corpus_id {cid}.")
            page_index = int(row["page_number_in_doc"]) - origin
            mapping[cid] = {
                "doc_id": doc["doc_id"],
                "doc_version": doc["id"],
                "page_index": page_index,
                "page_number": page_index + 1,
                "corpus_id": cid,
            }
    wanted, relevant = set(ids), defaultdict(dict)
    for row in qrels:
        qid, cid = str(row["query_id"]), str(row["corpus_id"])
        if qid not in wanted:
            continue
        if cid not in mapping:
            raise RagError(f"qrels references unknown corpus_id {cid}.")
        grade = int(row["score"])
        if grade not in (0, 1, 2):
            raise RagError(f"Unexpected qrel grade {grade}.")
        if grade > relevant[qid].get(cid, {}).get("grade", -1):
            relevant[qid][cid] = {
                **mapping[cid],
                "grade": grade,
                "content_type": row.get("content_type", []),
                "bounding_boxes": row.get("bounding_boxes", []),
            }
    if any(not any(p["grade"] > 0 for p in relevant[qid].values()) for qid in wanted):
        raise RagError("At least one English question has no positive relevance annotation.")
    # Keep query-only input physically separate from labels/answers for the runner.
    runtime = [{"query_id": str(q["query_id"]), "query": q["query"]} for q in english]
    labels = [
        {
            "query_id": str(q["query_id"]),
            "answer": q.get("answer"),
            "raw_answers": q.get("raw_answers", []),
            "content_type": q.get("content_type", []),
            "query_types": q.get("query_types", []),
            "query_format": q.get("query_format"),
            "qrels": list(relevant[str(q["query_id"])].values()),
        }
        for q in english
    ]
    return (
        runtime,
        labels,
        {
            "page_origins": origins,
            "page_map": mapping,
            "language": "english",
            "english_count": len(english),
        },
    )


def download(config: Config, store: Store) -> dict:
    """Publish an English-only dataset; annotation downloads are transient."""
    dest = (config.work_dir / "raw" / config.dataset.revision).resolve()
    try:
        return _download(config, store)
    finally:
        # Remove legacy raw annotation downloads too. Only explicitly named files
        # under this dataset directory are eligible; never recursively delete data.
        for folder in ("queries", "qrels"):
            for path in (dest / folder).glob("*.parquet"):
                if not path.resolve().is_relative_to(dest):
                    raise RagError("Refusing to clean an annotation path outside the dataset directory.")
                path.unlink()


def _verify_download(path: Path, remote, label: str) -> str:
    checksum = sha256(path)
    if remote.size is not None and remote.size != path.stat().st_size:
        raise RagError(f"Download size mismatch: {label}; remove only this cached file and download again.")
    if remote.lfs:
        expected = remote.lfs.get("sha256") if isinstance(remote.lfs, dict) else remote.lfs.sha256
        if checksum != expected:
            raise RagError(
                f"Download SHA-256 mismatch: {label}; cached file differs from the pinned benchmark."
            )
    elif remote.blob_id:
        git_hash = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                git_hash.update(block)
        if git_hash.hexdigest() != remote.blob_id:
            raise RagError(f"Download Git blob checksum mismatch: {label}")
    return checksum


def _download(config: Config, store: Store) -> dict:
    import pyarrow.parquet as pq
    import pymupdf
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    dest = config.work_dir / "raw" / config.dataset.revision
    patterns = [
        "pdfs/*.pdf",
        "pdfs/metadata.csv",
        "documents_metadata/*.parquet",
        "corpus/*.parquet",
    ]
    try:
        repo_info = HfApi().dataset_info(
            config.dataset.repo, revision=config.dataset.revision, files_metadata=True
        )
        if repo_info.sha != config.dataset.revision:
            raise RagError("HF returned a different dataset revision.")
        snapshot_download(
            repo_id=config.dataset.repo,
            repo_type="dataset",
            revision=config.dataset.revision,
            allow_patterns=patterns,
            local_dir=hub_local_dir(dest),
            max_workers=2,
        )
    except Exception as e:
        raise RagError(f"Dataset download failed ({type(e).__name__}); rerun to resume the HF cache.") from e

    # Verify cached downloads against the pinned upstream objects too, not only against a new local manifest.
    upstream = {r.rfilename: r for r in repo_info.siblings}
    checksums = {}
    for path in sorted(dest.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        rel = path.relative_to(dest).as_posix()
        if rel.startswith(("queries/", "qrels/")):
            continue
        if rel not in upstream:
            raise RagError(f"Unexpected file in the pinned dataset directory: {rel}")
        checksums[rel] = _verify_download(path, upstream[rel], rel)

    def table(folder: str, columns=None) -> list[dict]:
        files = sorted((dest / folder).glob("*.parquet"))
        if not files:
            raise RagError(f"Missing dataset parquet files: {folder}")
        return [r for f in files for r in pq.read_table(f, columns=columns).to_pylist()]

    metadata = table("documents_metadata")
    pdfs = {p.name: p for p in (dest / "pdfs").glob("*.pdf")}
    if set(pdfs) != set(PDF_PAGES) or len(metadata) != config.dataset.expected_documents:
        raise RagError("Expected the six original benchmark PDFs and six metadata rows.")
    documents = []
    for row in metadata:
        name = Path(row["file_name"]).name
        if name not in pdfs:
            raise RagError(f"Metadata filename is not a benchmark PDF: {name}")
        pdf = pdfs[name]
        with pymupdf.open(pdf) as opened:
            pages = len(opened)
        if pages != PDF_PAGES[name] or pages != int(row["page_number"]):
            raise RagError(f"PDF pagination differs from benchmark metadata: {name} has {pages} pages.")
        checksum = sha256(pdf)
        doc = {
            "id": digest(
                {"doc_id": str(row["doc_id"]), "sha256": checksum, "revision": config.dataset.revision}
            ),
            "doc_id": str(row["doc_id"]),
            "filename": name,
            "pdf_path": str(pdf.resolve()),
            "sha256": checksum,
            "pages": pages,
            "metadata": row,
            "dataset_revision": config.dataset.revision,
        }
        documents.append(doc)
    if len({d["doc_id"] for d in documents}) != len(documents):
        raise RagError("Duplicate document IDs in dataset metadata.")
    if sum(d["pages"] for d in documents) != config.dataset.expected_pages:
        raise RagError("Unexpected total page count.")
    annotation_checksums = {}
    english, english_qrels = [], []
    # The upstream annotation files are inspected in a temporary local directory.
    # Only English rows leave this scope; no translated QA/qrels are persisted.
    with TemporaryDirectory(prefix="finance-english-", dir=config.work_dir.resolve()) as temporary:
        if not Path(temporary).resolve().is_relative_to(config.work_dir.resolve()):
            raise RagError("Temporary annotation directory is outside the workspace.")
        for folder in ("queries", "qrels"):
            files = sorted(p for p in upstream if p.startswith(folder + "/") and p.endswith(".parquet"))
            if not files:
                raise RagError(f"Missing upstream {folder} parquet files.")
            wanted = {str(q["query_id"]) for q in english}
            for name in files:
                try:
                    downloaded = Path(
                        hf_hub_download(
                            config.dataset.repo,
                            name,
                            repo_type="dataset",
                            revision=config.dataset.revision,
                            local_dir=hub_local_dir(Path(temporary)),
                        )
                    )
                except Exception as e:
                    raise RagError(
                        f"English annotation download failed ({type(e).__name__}); rerun download."
                    ) from e
                annotation_checksums[name] = _verify_download(downloaded, upstream[name], name)
                rows = pq.read_table(downloaded).to_pylist()
                if folder == "queries":
                    english.extend(r for r in rows if str(r["language"]).strip().lower() == "english")
                else:
                    english_qrels.extend(r for r in rows if str(r["query_id"]) in wanted)
    runtime, labels, audit = prepare_annotations(
        english,
        english_qrels,
        table("corpus", ["corpus_id", "doc_id", "page_number_in_doc"]),
        documents,
        config.dataset.expected_english_queries,
    )
    manifest = {
        "repo": config.dataset.repo,
        "revision": config.dataset.revision,
        "synthetic": False,
        "language": "english",
        "annotation_policy": "english_only_transient_upstream_downloads",
        "annotation_source_sha256": annotation_checksums,
        "documents": documents,
        "counts": {
            "documents": len(documents),
            "pages": sum(d["pages"] for d in documents),
            "queries": len(runtime),
        },
        "files": checksums,
    }
    manifest["id"] = manifest_identity(manifest)
    prepared = config.work_dir / "datasets" / manifest["id"]
    write_jsonl(prepared / "queries.jsonl", runtime)
    write_jsonl(prepared / "labels.jsonl", labels)
    write_json(prepared / "mapping.json", audit)
    manifest["prepared_dir"] = str(prepared)
    manifest["queries_sha256"] = sha256(prepared / "queries.jsonl")
    manifest["labels_sha256"] = sha256(prepared / "labels.jsonl")
    write_json(prepared / "manifest.json", store.portable(manifest))
    for doc in documents:
        store.put("documents", doc["id"], doc)
    store.set_meta("dataset", manifest)
    return manifest


def manifest_identity(manifest: dict) -> str:
    content = {
        k: v
        for k, v in manifest.items()
        if k not in {"id", "prepared_dir", "queries_sha256", "labels_sha256"}
    }
    content["documents"] = [{k: v for k, v in d.items() if k != "pdf_path"} for d in manifest["documents"]]
    return digest(content)


def dataset(store: Store, verify_pdfs: bool = False, verify_queries: bool = False) -> dict:
    manifest = store.meta("dataset")
    if not manifest and (release := store.meta("fact_release")):
        # A fact release is usable independently of benchmark questions and labels.
        docs = [
            store.get("documents", row[0]) for row in store.db.execute("SELECT id FROM documents ORDER BY id")
        ]
        manifest = {
            "id": digest(["document-only-v1", release["release_id"], [d["id"] for d in docs]]),
            "documents": docs,
            "synthetic": any(
                d.get("synthetic", False) or d.get("dataset_revision", "").startswith("fixture-")
                for d in docs
            ),
            "document_only": True,
            "fact_release_id": release["release_id"],
        }
    if not manifest:
        raise RagError("No prepared dataset. Run download (or fixture for offline verification).")
    # Document-only stages do not even open the query/label files. Evaluation validates labels itself.
    if verify_queries:
        if manifest.get("document_only"):
            raise RagError(
                "This is a document-only fact/retrieval release; attach validated English benchmark data before evaluation."
            )
        queries = Path(manifest["prepared_dir"]) / "queries.jsonl"
        if not queries.exists() or sha256(queries) != manifest["queries_sha256"]:
            raise RagError("Prepared queries.jsonl has changed; prepare a new dataset version.")
    if verify_pdfs:
        for doc in manifest["documents"]:
            verify_document(doc)
    return manifest


def verify_document(doc: dict):
    pdf = Path(doc["pdf_path"])
    if not pdf.exists() or sha256(pdf) != doc["sha256"]:
        raise RagError(f"Raw PDF missing or modified: {doc['filename']}; source version cannot be verified.")


def inspect(config: Config, store: Store, tables: bool = False) -> dict:
    import pymupdf

    manifest = dataset(store, verify_pdfs=True)
    report = {"dataset": manifest["id"], "documents": [], "tables_requested": tables}
    for doc in manifest["documents"]:
        pages = []
        with pymupdf.open(doc["pdf_path"]) as pdf:
            for i, page in enumerate(pdf):
                info = {"page_index": i, "page_number": i + 1}
                try:
                    text = page.get_text("text")
                    info.update(
                        text_characters=len(text),
                        non_whitespace_characters=len("".join(text.split())),
                        empty_text=not bool(text.strip()),
                        replacement_characters=text.count("\ufffd"),
                        unexpected_controls=sum(ord(c) < 32 and c not in "\n\r\t" for c in text),
                        normalized_text_sha256=hashlib.sha256(" ".join(text.split()).encode()).hexdigest(),
                        embedded_images=len(page.get_images()),
                        drawings=len(page.get_drawings()),
                        width=page.rect.width,
                        height=page.rect.height,
                        native_tables=len(page.find_tables().tables) if tables else None,
                    )
                except Exception as e:
                    info["error"] = f"{type(e).__name__}: {e}"
                pages.append(info)
        parses = []
        for kind in ("flat", "structured"):
            pid = store.meta(f"parse:{kind}:{doc['id']}")
            if pid:
                parsed = store.get("parses", pid)
                nodes = store.nodes(pid)
                parses.append(
                    {
                        "kind": kind,
                        "parse_id": pid,
                        "status": parsed["status"],
                        "snapshot": parsed.get("snapshot"),
                        "failed_pages": parsed["failed_pages"],
                        "errors": parsed.get("errors", []),
                        "nodes": len(nodes),
                        "node_types": dict(Counter(n["kind"] for n in nodes)),
                        "nodes_without_source": [n["ref"] for n in nodes if not n["sources"]],
                        "nodes_without_bbox": [
                            n["ref"]
                            for n in nodes
                            if n["sources"] and not any(s.get("bbox") for s in n["sources"])
                        ],
                    }
                )
        report["documents"].append(
            {
                "doc_id": doc["doc_id"],
                "doc_version": doc["id"],
                "sha256": doc["sha256"],
                "pages": pages,
                "parses": parses,
            }
        )
    path = config.work_dir / "inspection.json"
    write_json(path, report)
    from .diagnostics.quality import quality_report

    quality = quality_report(config, store, report)
    return {
        "report": str(path),
        "quality_report": quality["report"],
        "quality_csv": quality["csv"],
        "documents": len(report["documents"]),
        "empty_text_pages": sum(p.get("empty_text", False) for d in report["documents"] for p in d["pages"]),
        "failed_pages": sum("error" in p for d in report["documents"] for p in d["pages"]),
    }
