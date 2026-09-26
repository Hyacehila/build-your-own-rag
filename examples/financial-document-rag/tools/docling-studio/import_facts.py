"""Import frozen facts into an isolated, unmodified Docling Studio installation.

Run with Studio's Python environment. No conversion, model, or embedding calls.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def main(args):
    source = args.facts.resolve()
    output = args.output.resolve()
    backend = args.studio.resolve() / "document-parser"
    output.mkdir(parents=True, exist_ok=True)
    database = output / "studio.sqlite"
    os.environ["DB_PATH"] = str(database)
    sys.path.insert(0, str(backend))

    from docling_core.types.doc import DoclingDocument
    from domain.models import AnalysisJob, Chunk, Document
    from domain.value_objects import ChunkBbox, ChunkDocItem, DocumentLifecycleState
    from persistence.analysis_repo import SqliteAnalysisRepository
    from persistence.chunk_repo import SqliteChunkRepository
    from persistence.database import init_db
    from persistence.document_repo import SqliteDocumentRepository
    from services.analysis_service import _chunk_to_dict

    db = sqlite3.connect((source / "experiment.sqlite").as_uri() + "?mode=ro", uri=True)
    release = json.loads(db.execute("SELECT value FROM metadata WHERE key='fact_release'").fetchone()[0])
    if release.get("requires_final_outline_review"):
        acceptance_row = db.execute(
            "SELECT value FROM metadata WHERE key='outline_acceptance'"
        ).fetchone()
        if not acceptance_row or release.get("publication_status") != "accepted":
            raise ValueError("Candidate facts need final outline acceptance before Studio import.")
        acceptance = json.loads(acceptance_row[0])
        if acceptance.get("release_id") != release["release_id"]:
            raise ValueError("Final outline acceptance is bound to another fact release.")
        expected = {row["doc_id"] for row in release["documents"]}
        reviewed = {row.get("doc_id") for row in acceptance.get("documents", [])}
        if reviewed != expected:
            raise ValueError("Final outline acceptance does not cover every document.")
    receipt_path = output / "import.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["release_id"] != release["release_id"] or not database.exists():
            raise ValueError("Use a new output directory for a different release or missing database.")
        print("Already imported this release; preserving Studio edits.")
        return
    if database.exists():
        raise ValueError("Partial or foreign Studio database: choose a new output directory.")

    def get(table, key):
        return json.loads(db.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()[0])

    def resource(locator):
        path = (source / locator).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Source resource escapes fact directory.")
        return path

    await init_db()
    docs, analyses, chunks = SqliteDocumentRepository(), SqliteAnalysisRepository(), SqliteChunkRepository()
    receipt = {"release_id": release["release_id"], "conversion_calls": 0, "documents": []}
    for row in release["documents"]:
        parsed = get("parses", row["parse_id"])
        document = get("documents", parsed["doc_version"])
        snapshot = resource(parsed["snapshot"])
        pdf = resource(document["pdf_path"])
        if checksum(snapshot) != parsed["snapshot_sha256"] or checksum(pdf) != document["sha256"]:
            raise ValueError("Frozen source checksum mismatch.")
        raw = snapshot.read_bytes().decode("utf-8")
        docling = DoclingDocument.model_validate_json(raw)
        nodes = {
            n["id"]: n
            for (payload,) in db.execute("SELECT payload FROM nodes WHERE parse_id=?", (parsed["id"],))
            for n in [json.loads(payload)]
        }
        pieces = json.loads(
            (source / "diagnostics" / row["doc_id"] / "chunks.json").read_text(encoding="utf-8")
        )
        if any(nid not in nodes for piece in pieces for nid in piece["node_ids"]):
            raise ValueError("Diagnostic chunk references another parse version.")
        doc_id = row["doc_id"] + "-" + parsed["id"][:12]
        copied = output / "pdfs" / (doc_id + ".pdf")
        copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(pdf, copied)
        pages = {
            number: {
                "page_number": number,
                "width": page.size.width,
                "height": page.size.height,
                "elements": [],
            }
            for number, page in docling.pages.items()
        }
        # Original page numbers and already-normalized TOPLEFT bounds are preserved.
        # Only adapt field names to Studio's PageDetail contract; no inferred parents.
        for node in nodes.values():
            if node["reading_order"] is None:
                continue
            for loc in node["sources"]:
                pages[loc["page_number"]]["elements"].append(
                    {
                        "type": node["kind"],
                        "bbox": loc.get("bbox") or [0, 0, 0, 0],
                        "content": node["text"],
                        "level": node["depth"],
                        "self_ref": node["ref"],
                    }
                )
        doc = Document(
            id=doc_id,
            filename=document["filename"].replace(".pdf", f" [{release['contract']}].pdf"),
            content_type="application/pdf",
            file_size=copied.stat().st_size,
            page_count=document["pages"],
            storage_path=str(copied),
            lifecycle_state=DocumentLifecycleState.CHUNKED,
        )
        await docs.insert(doc)
        job = AnalysisJob(id=parsed["id"], document_id=doc.id)
        await analyses.insert(job)
        job.mark_running()
        job.mark_completed(
            markdown=docling.export_to_markdown(),
            html="",
            pages_json=json.dumps([pages[p] for p in sorted(pages)], ensure_ascii=False),
            document_json=raw,
        )
        await analyses.update_status(job)
        mapped = [
            Chunk(
                document_id=doc.id,
                sequence=i,
                text=c["text"],
                headings=c["headings"],
                source_page=c["sources"][0]["page_number"] if c["sources"] else None,
                bboxes=[
                    ChunkBbox(page=s["page_number"], bbox=s["bbox"]) for s in c["sources"] if s.get("bbox")
                ],
                doc_items=[
                    ChunkDocItem(self_ref=nodes[n]["ref"], label=nodes[n]["kind"]) for n in c["node_ids"]
                ],
            )
            for i, c in enumerate(pieces)
        ]
        await chunks.insert_many(mapped)
        await analyses.update_chunks(job.id, json.dumps([_chunk_to_dict(c) for c in mapped]))
        receipt["documents"].append(
            {
                "doc_id": row["doc_id"],
                "studio_id": doc.id,
                "analysis_id": job.id,
                "snapshot_sha256": parsed["snapshot_sha256"],
                "pdf_sha256": document["sha256"],
                "diagnostic_chunks": len(mapped),
                "source": "imported; no new analysis executed",
            }
        )
        print(f"Imported {row['doc_id']}: {len(mapped)} diagnostic chunks", flush=True)
    db.close()
    with sqlite3.connect(database) as target:
        assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not target.execute("PRAGMA foreign_key_check").fetchall()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
