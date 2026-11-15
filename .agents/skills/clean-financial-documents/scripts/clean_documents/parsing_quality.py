"""Local diagnostics, not an accuracy score or permission to rewrite facts."""

from collections import Counter

import pymupdf

from .text_checks import lexical, numbers, recovery

QUALITY_CONTRACT = "source-table-span-diagnostics-v1"


def inspect_snapshot(snapshot, pdf_path):
    issues, tables = [], []
    for item in snapshot.get("texts", []):
        original = item.get("orig", item.get("text", ""))
        for prov in item.get("prov", []):
            start, end = prov["charspan"]
            if not 0 <= start <= end <= len(original):
                issues.append(
                    {
                        "check": "invalid_original_charspan",
                        "severity": "error",
                        "ref": item["self_ref"],
                        "page": prov["page_no"],
                        "charspan": [start, end],
                        "original_length": len(original),
                    }
                )
    with pymupdf.open(pdf_path) as pdf:
        for item in snapshot.get("tables", []):
            data = item["data"]
            text = "\n".join(c.get("text", "") for c in data["table_cells"])
            row = {
                "ref": item["self_ref"],
                "rows": data["num_rows"],
                "columns": data["num_cols"],
                "pages": [p["page_no"] for p in item["prov"]],
            }
            native = []
            for prov in item["prov"]:
                page = pdf[prov["page_no"] - 1]
                b = prov["bbox"]
                rect = pymupdf.Rect(b["l"], b["t"], b["r"], b["b"])
                if b["coord_origin"] == "BOTTOMLEFT":
                    rect = pymupdf.Rect(b["l"], page.rect.height - b["t"], b["r"], page.rect.height - b["b"])
                native.append(page.get_text("text", clip=rect, sort=True))
            native_text = "\n".join(native)
            expected, observed = lexical(native_text), lexical(text)
            row.update(
                native_terms=sum(expected.values()),
                term_recovery=recovery(expected, observed),
                number_recovery=recovery(numbers(native_text), numbers(text)),
                missing_terms=dict(expected - observed),
                missing_numbers=dict(numbers(native_text) - numbers(text)),
            )
            checks = []
            if not text.strip():
                checks.append("empty_table_needs_image_route")
            elif sum(expected.values()) >= 20 and row["term_recovery"] < 0.95:
                checks.append("table_region_text_not_assigned")
            if row["missing_numbers"]:
                checks.append("table_region_numbers_not_assigned")
            if data["table_cells"] and not any(c.get("column_header") for c in data["table_cells"]):
                checks.append("table_without_declared_header")
            for check in checks:
                issues.append(
                    {"check": check, "severity": "candidate", "ref": item["self_ref"], "pages": row["pages"]}
                )
            tables.append(row)
    return {
        "contract": QUALITY_CONTRACT,
        "candidate_only": True,
        "limitations": "Native PDF text can itself be wrong; region overlap is not table accuracy. "
        "Flags require a bounded alternate parse or source-image verification, never automatic cell edits.",
        "issues": issues,
        "counts": dict(Counter(i["check"] for i in issues)),
        "tables": tables,
    }
