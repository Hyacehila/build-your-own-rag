"""Page-wide coverage diagnostics, separate from semantic accuracy."""

from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from ..common import write_json
from ..storage import Store
from .text_checks import lexical, recovery


def page_terms(nodes):
    output = defaultdict(Counter)
    for node in nodes:
        if node.get("reading_order") is None:
            continue
        raw = node.get("raw", {})
        if raw.get("content_layer", "body") != "body":
            continue
        text = node["text"]
        if "table_cells" in raw.get("data", {}):
            text = "\n".join(c["text"] for c in raw["data"]["table_cells"])
        for s in node["sources"]:
            piece = text
            if len(node["sources"]) > 1 and raw.get("orig"):
                prov = [p for p in raw.get("prov", []) if p["page_no"] == s["page_number"]]
                if len(prov) == 1:
                    a, b = prov[0]["charspan"]
                    if 0 <= a <= b <= len(raw["orig"]):
                        piece = raw["orig"][a:b]
            output[s["page_number"]].update(lexical(piece))
    return output


def audit_pages(root):
    reports = []
    with Store(root, read_only=True) as store:
        release = store.meta("fact_release")
        for row in release["documents"]:
            parsed = store.get("parses", row["parse_id"])
            doc = store.get("documents", parsed["doc_version"])
            nodes = store.nodes(parsed["id"])
            cleaned = page_terms(nodes)
            original = page_terms(store.nodes(row["raw_parse_id"]))
            assets = {
                s["page_number"] for n in nodes if n["kind"] in {"picture", "table"} for s in n["sources"]
            }
            excluded = set(row["excluded_pages"])
            pages = []
            with pymupdf.open(doc["pdf_path"]) as pdf:
                for page in pdf:
                    number = page.number + 1
                    if number in excluded:
                        continue
                    expected = lexical(page.get_text("text", sort=True))
                    pages.append(
                        {
                            "page": number,
                            "native_terms": sum(expected.values()),
                            "raw_body_recovery": recovery(expected, original[number]),
                            "clean_body_recovery": recovery(expected, cleaned[number]),
                            "has_asset": number in assets,
                        }
                    )
            low = [p for p in pages if p["native_terms"] >= 100 and p["clean_body_recovery"] < 0.8]
            regressions = [
                p
                for p in pages
                if p["native_terms"] >= 100 and p["raw_body_recovery"] - p["clean_body_recovery"] > 0.1
            ]
            report = {
                "doc_id": row["doc_id"],
                "pages_checked": len(pages),
                "low_coverage_pages": low,
                "coverage_drop_over_10pct": regressions,
                "pages": pages,
                "limitation": "Native-term overlap is a diagnostic proxy, not OCR or answer accuracy; diagram text may intentionally remain in source images.",
            }
            write_json(root / "diagnostics" / row["doc_id"] / "page-quality.json", report)
            reports.append({k: v for k, v in report.items() if k != "pages"})
    write_json(root / "page-quality.json", reports)
    return reports


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/facts-v2-release"))
    args = parser.parse_args(argv)
    for r in audit_pages(args.output):
        print(
            r["doc_id"],
            "low coverage:",
            len(r["low_coverage_pages"]),
            "new drops:",
            len(r["coverage_drop_over_10pct"]),
        )


if __name__ == "__main__":
    main()
