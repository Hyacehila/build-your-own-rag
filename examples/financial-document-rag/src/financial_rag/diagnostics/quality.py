"""Source-only cleaning audit; findings never rewrite PDFs, questions or answers."""

import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from ..common import write_json
from ..config import Config
from ..dataset import dataset
from ..storage import Store


def inspect_nodes(nodes: list[dict], document: dict, sizes: dict[int, dict]) -> dict:
    counts = Counter(n["kind"] for n in nodes)
    issues = defaultdict(list)
    for node in nodes:
        ref = node["ref"]
        if (node["text"].strip() or node["kind"] in {"table", "picture"}) and not node["sources"]:
            issues["content_without_source"].append(ref)
        for source in node["sources"]:
            page = source["page_index"]
            if (
                source["doc_version"] != document["doc_version"]
                or source["doc_id"] != document["doc_id"]
                or source["page_number"] != page + 1
                or page not in sizes
            ):
                issues["invalid_page_binding"].append(ref)
                continue
            box = source.get("bbox")
            if box is None:
                issues["source_without_bbox"].append(ref)
            elif (
                len(box) != 4
                or not all(math.isfinite(x) for x in box)
                or not -1 <= box[0] <= box[2] <= sizes[page]["width"] + 1
                or not -1 <= box[1] <= box[3] <= sizes[page]["height"] + 1
            ):
                issues["invalid_bbox"].append(ref)
        if node["kind"] == "table":
            cells = node.get("raw", {}).get("data", {}).get("table_cells", [])
            if not cells or not any(c.get("text", "").strip() for c in cells):
                issues["empty_table"].append(ref)
            if not any(c.get("column_header") for c in cells):
                issues["table_without_column_header_flag"].append(ref)
        elif node["kind"] == "picture" and not node.get("caption", "").strip():
            issues["picture_without_original_caption"].append(ref)
    return {"node_types": dict(counts), "issues": {k: sorted(set(v)) for k, v in issues.items()}}


def quality_report(config: Config, store: Store, inspection: dict) -> dict:
    manifest = dataset(store)
    rows, details, duplicate_pages = [], [], defaultdict(list)
    for doc in inspection["documents"]:
        pages = doc["pages"]
        row = {
            "document": doc["doc_id"],
            "pages": len(pages),
            "empty_text_pages": sum(p.get("empty_text", False) for p in pages),
            "sparse_text_pages": sum(0 < p.get("non_whitespace_characters", 0) < 80 for p in pages),
            "image_pages": sum(p.get("embedded_images", 0) > 0 for p in pages),
            "image_only_pages": sum(
                p.get("empty_text", False) and p.get("embedded_images", 0) > 0 for p in pages
            ),
            "replacement_characters": sum(p.get("replacement_characters", 0) for p in pages),
            "unexpected_controls": sum(p.get("unexpected_controls", 0) for p in pages),
            "native_inspection_failures": sum("error" in p for p in pages),
            "flat_status": "not_parsed",
            "structured_status": "not_parsed",
            "tables": 0,
            "pictures": 0,
            "missing_table_headers": 0,
            "empty_tables": 0,
            "content_without_source": 0,
            "invalid_page_binding": 0,
            "invalid_bbox": 0,
        }
        for page in pages:
            if page.get("non_whitespace_characters", 0) >= 128:
                duplicate_pages[page["normalized_text_sha256"]].append(
                    {"doc_id": doc["doc_id"], "page_number": page["page_number"]}
                )
        parsed_details = []
        for parsed in doc["parses"]:
            row[parsed["kind"] + "_status"] = parsed["status"]
            if parsed["kind"] == "structured":
                audit = inspect_nodes(
                    store.nodes(parsed["parse_id"]), doc, {p["page_index"]: p for p in pages}
                )
                parsed_details.append({"parse_id": parsed["parse_id"], **audit})
                row["tables"] = audit["node_types"].get("table", 0)
                row["pictures"] = audit["node_types"].get("picture", 0)
                for target, key in (
                    ("missing_table_headers", "table_without_column_header_flag"),
                    ("empty_tables", "empty_table"),
                    ("content_without_source", "content_without_source"),
                    ("invalid_page_binding", "invalid_page_binding"),
                    ("invalid_bbox", "invalid_bbox"),
                ):
                    row[target] = len(audit["issues"].get(key, []))
        details.append(
            {
                "doc_id": doc["doc_id"],
                "empty_text_pages": [p["page_number"] for p in pages if p.get("empty_text")],
                "sparse_text_pages": [
                    p["page_number"] for p in pages if 0 < p.get("non_whitespace_characters", 0) < 80
                ],
                "parses": parsed_details,
            }
        )
        rows.append(row)
    duplicates = [v for v in duplicate_pages.values() if len(v) > 1]
    root = config.work_dir / "inspection"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "quality.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    json_path = root / "quality.json"
    write_json(
        json_path,
        {
            "dataset_id": manifest["id"],
            "revision": manifest["revision"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "documents": details,
            "summary": rows,
            "identical_native_text_groups": duplicates,
            "policy": "Keep all original physical pages and values; diagnostics are not benchmark scores.",
        },
    )
    report = root / "quality.md"
    lines = [
        "# 数据清洗与解析检查",
        "",
        f"数据版本：`{manifest['revision']}`。英语问题：{manifest['counts']['queries']}。",
        "",
        "原始 PDF 和物理页序保持完整；空白页、重复页均保留，以维持 benchmark 来源映射。",
        "此报告只检查文档，不使用标准问题、答案或证据标注来修改解析。",
        "",
        "| 文档 | 页数 | 无正文页 | 普通解析 | 结构解析 | 表格 | 图片 | 无表头标记 | 来源/坐标异常 |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        defects = r["content_without_source"] + r["invalid_page_binding"] + r["invalid_bbox"]
        lines.append(
            f"| {r['document']} | {r['pages']} | {r['empty_text_pages']} | {r['flat_status']} | "
            f"{r['structured_status']} | {r['tables']} | {r['pictures']} | {r['missing_table_headers']} | {defects} |"
        )
    lines += [
        "",
        "检查项的解释：",
        "",
        "- 无正文页指去除空白后没有本地文本，需结合页图检查；不自动当作解析失败或删除。",
        "- 无表头标记表示 Docling 未将任何单元格标为列标题，需要抽查，不自动补写标题。",
        "- 图片没有原始 caption 时不编造 caption；C 的图片描述会单独记录生成来源。",
        "- 替代字符、控制字符、稀疏文本、空表和来源异常详见配套 JSON/CSV；不自动改写财务数值。",
        f"- 发现 {len(duplicates)} 组规范空白后文本相同的页面；仅作为审查线索，仍保留不同页面的 corpus ID。",
        "",
        "逐页统计见上级 `inspection.json`，节点引用及异常见 `quality.json`。",
        "这些检查不能证明每个表格单元格或图表都解析正确，仍需人工抽查与实验误差分析。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(report), "json": str(json_path), "csv": str(csv_path)}
