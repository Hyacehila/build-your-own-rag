"""Original, fictional three-page test document; never a benchmark substitute."""

from __future__ import annotations

from io import BytesIO

from .common import digest, sha256, write_json, write_jsonl
from .config import Config
from .parsing import docling_nodes, parse_signature
from .storage import Store


def create_fixture(config: Config, store: Store, structured_snapshot: bool = False) -> dict:
    import pymupdf
    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )
    from PIL import Image, ImageDraw

    root = config.work_dir / "fixture"
    root.mkdir(parents=True, exist_ok=True)
    pdf_path = root / "fictional-bank.pdf"
    text1 = (
        "Example Bank is a fictional company used only for software verification. "
        "Revenue increased from USD 120 million in 2023 to USD 150 million in 2024. "
        "The increase was USD 30 million, or 25 percent. This is not a real financial report."
    )
    text3 = (
        "Risk and liquidity. Example Bank held USD 42 million in cash at year end 2024. "
        "The table on the previous page reports revenue by fictional business unit. "
        "All figures and answers in this fixture are invented for offline software tests."
    )
    rows = [
        ["Business unit", "2023 USD m", "2024 USD m"],
        ["Retail", "80", "100"],
        ["Advisory", "40", "50"],
        ["Total", "120", "150"],
    ]
    # Extra fictional rows force header-preserving table splitting under small test budgets.
    rows.extend([[f"Test segment {i:02d}", str(i), str(i + 1)] for i in range(1, 25)])
    chart = Image.new("RGB", (720, 330), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((22, 10), "FICTIONAL REVENUE (USD million)", fill="black", font_size=22)
    draw.line((70, 270, 660, 270), fill="black", width=3)
    draw.rectangle((180, 102, 310, 268), fill="#4989be")
    draw.rectangle((420, 60, 550, 268), fill="#2e7159")
    for xy, label in [((210, 282), "2023"), ((450, 282), "2024"), ((210, 77), "120"), ((450, 35), "150")]:
        draw.text(xy, label, fill="black", font_size=22)
    image = BytesIO()
    chart.save(image, format="PNG")
    with pymupdf.open() as pdf:
        first = pdf.new_page(width=612, height=792)
        first.insert_text((40, 42), "FICTIONAL DATA - OFFLINE TEST ONLY", fontsize=17)
        first.insert_text((40, 85), "1. Performance", fontsize=16)
        first.insert_textbox((40, 105, 572, 220), text1, fontsize=12)
        first.insert_image((40, 245, 572, 490), stream=image.getvalue())
        first.insert_text(
            (40, 516), "Figure 1. Fictional revenue in USD million, 2023 and 2024.", fontsize=11
        )
        second = pdf.new_page(width=612, height=792)
        second.insert_text((40, 42), "1.1 Revenue by business unit", fontsize=16)
        second.insert_text((40, 68), "Table 1. Fictional business unit revenue (USD million).", fontsize=11)
        for i, row in enumerate(rows):
            y = 88 + i * 22
            for j, value in enumerate(row):
                x0, x1 = [40, 300, 430, 572][j : j + 2]
                second.draw_rect((x0, y, x1, y + 22), color=(0.5, 0.5, 0.5))
                second.insert_text((x0 + 5, y + 15), value, fontsize=10)
        third = pdf.new_page(width=612, height=792)
        third.insert_text((40, 42), "2. Risk and liquidity", fontsize=16)
        third.insert_textbox((40, 80, 572, 220), text3, fontsize=12)
        for i, page in enumerate(pdf):
            page.insert_text((40, 760), f"Fictional fixture - physical page {i + 1}", fontsize=9)
        pdf.save(pdf_path, no_new_id=True)
    checksum = sha256(pdf_path)
    doc = {
        "id": digest(["fixture-v1", checksum]),
        "doc_id": "fictional-bank",
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "sha256": checksum,
        "pages": 3,
        "dataset_revision": "fixture-v1",
    }
    store.put("documents", doc["id"], doc)
    queries = [
        {"query_id": "fixture-revenue", "query": "What was Example Bank's revenue in 2024 in USD million?"},
        {"query_id": "fixture-growth", "query": "What was the revenue growth percentage from 2023 to 2024?"},
        {"query_id": "fixture-cash", "query": "How much cash did Example Bank hold at year end 2024?"},
    ]
    labels = []
    for q, answer, pages, content in zip(
        queries,
        ["USD 150 million", "25 percent", "USD 42 million"],
        [[0, 1], [0], [2]],
        [["text", "table"], ["chart"], ["text"]],
    ):
        labels.append(
            {
                "query_id": q["query_id"],
                "answer": answer,
                "raw_answers": [],
                "content_type": content,
                "query_types": ["numeric"],
                "query_format": "short_answer",
                "qrels": [
                    {
                        "doc_id": doc["doc_id"],
                        "doc_version": doc["id"],
                        "page_index": p,
                        "page_number": p + 1,
                        "grade": 2,
                    }
                    for p in pages
                ],
            }
        )
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "labels.jsonl", labels)
    manifest = {
        "id": digest(["fixture-v1", checksum]),
        "repo": "original-fictional-fixture",
        "revision": "fixture-v1",
        "synthetic": True,
        "language": "english",
        "documents": [doc],
        "counts": {"documents": 1, "pages": 3, "queries": 3},
        "prepared_dir": str(root),
        "queries_sha256": sha256(root / "queries.jsonl"),
        "labels_sha256": sha256(root / "labels.jsonl"),
    }
    write_json(root / "manifest.json", manifest)
    store.set_meta("dataset", manifest)
    if structured_snapshot:
        # This known-layout document validates DoclingDocument/HybridChunker offline;
        # it is explicitly not a result of running the learned PDF layout parser.
        dl_doc = DoclingDocument(name="fictional-bank")
        for p in range(1, 4):
            dl_doc.add_page(page_no=p, size=Size(width=612, height=792))

        def prov(page, box, text=""):
            return ProvenanceItem(
                page_no=page,
                bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
                charspan=(0, len(text)),
            )

        h1 = dl_doc.add_heading(text="1. Performance", level=1, prov=prov(1, [40, 65, 572, 90]))
        dl_doc.add_text(
            label=DocItemLabel.TEXT, text=text1, parent=h1, prov=prov(1, [40, 105, 572, 220], text1)
        )
        caption = dl_doc.add_text(
            label=DocItemLabel.CAPTION,
            text="Figure 1. Fictional revenue in USD million, 2023 and 2024.",
            parent=h1,
            prov=prov(1, [40, 500, 572, 522]),
        )
        dl_doc.add_picture(caption=caption, parent=h1, prov=prov(1, [40, 245, 572, 490]))
        h2 = dl_doc.add_heading(
            text="1.1 Revenue by business unit", level=2, parent=h1, prov=prov(2, [40, 25, 572, 49])
        )
        cap = dl_doc.add_text(
            label=DocItemLabel.CAPTION,
            text="Table 1. Fictional business unit revenue (USD million).",
            parent=h2,
            prov=prov(2, [40, 54, 572, 74]),
        )
        cells = [
            TableCell(
                text=value,
                start_row_offset_idx=i,
                end_row_offset_idx=i + 1,
                start_col_offset_idx=j,
                end_col_offset_idx=j + 1,
                column_header=(i == 0),
            )
            for i, row in enumerate(rows)
            for j, value in enumerate(row)
        ]
        dl_doc.add_table(
            data=TableData(num_rows=len(rows), num_cols=3, table_cells=cells),
            caption=cap,
            parent=h2,
            prov=prov(2, [40, 88, 572, 88 + len(rows) * 22]),
        )
        h3 = dl_doc.add_heading(text="2. Risk and liquidity", level=1, prov=prov(3, [40, 25, 572, 50]))
        dl_doc.add_text(
            label=DocItemLabel.TEXT, text=text3, parent=h3, prov=prov(3, [40, 80, 572, 220], text3)
        )
        signature = parse_signature(config, "structured")
        pid = digest({"doc_version": doc["id"], "signature": signature, "fixture": True})
        snapshot = config.work_dir / "parses" / pid / "document.json"
        write_json(snapshot, dl_doc.model_dump(mode="json"))
        parsed = {
            "id": pid,
            "doc_version": doc["id"],
            "kind": "structured",
            "signature": signature,
            "fixture": True,
            "status": "success",
            "failed_pages": [],
            "errors": [],
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256(snapshot),
        }
        store.put_parse(parsed, docling_nodes(dl_doc, doc, pid))
        store.set_meta(f"parse:structured:{doc['id']}", pid)
    return {
        "synthetic": True,
        "pdf": str(pdf_path),
        "structured_snapshot": structured_snapshot,
        "note": "Original fictional fixture. Not ViDoRe data; not benchmark results.",
    }
