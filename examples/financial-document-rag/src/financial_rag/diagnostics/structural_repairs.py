"""Deterministic, versioned candidates. Never overwrite the raw parser snapshot."""

import re
from collections import Counter

import pymupdf
from docling_core.types.doc import DocItemLabel, RefItem, SectionHeaderItem, TextItem, TitleItem

from ..common import RagError, digest

CONTRACT = "financial-boundaries-and-native-grid-v1"
ABBREVIATION = re.compile(r"^(?:[A-Z]\.){2,}(?:\s|$)")
EXHIBIT = re.compile(r"^Exhibit\s+\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))?\.?$", re.I)
NOTE = re.compile(r"^Note\s+(\d+)\b", re.I)
TABLE = re.compile(r"^Table\s+(\d+)\.\d+\b", re.I)


def semantic_content(doc):
    """Ignore structural classification only; text/data/provenance must stay exact."""

    def strip(value):
        if isinstance(value, list):
            return [strip(x) for x in value]
        if isinstance(value, dict):
            excluded = {"parent", "children", "level", "label"} if "self_ref" in value else set()
            return {k: strip(v) for k, v in value.items() if k not in excluded}
        return value

    return strip(doc.model_dump(mode="json"))


def _linear_tree(doc, heading_refs):
    """Reparent reading-flow items; preserve inline, list and asset subtrees.

    Only former/current heading containers are flattened. A table's own children
    and captions are never treated as document-level headings.
    """
    flow = []

    def visit(item):
        flow.append(item)
        if item.self_ref in heading_refs:
            retained = []
            children = list(item.children)
            for ref in children:
                child = ref.resolve(doc)
                if getattr(child, "label", None) == "inline":
                    retained.append(ref)
                else:
                    visit(child)
            item.children = retained

    for ref in list(doc.body.children):
        visit(ref.resolve(doc))
    doc.body.children = []
    stack = [(0, doc.body)]
    for item in flow:
        level = (
            item.level if isinstance(item, SectionHeaderItem) else 1 if isinstance(item, TitleItem) else None
        )
        if level is not None:
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()
        parent = stack[-1][1]
        item.parent = RefItem(cref=parent.self_ref)
        parent.children.append(RefItem(cref=item.self_ref))
        if level is not None:
            stack.append((level, item))


def repair_headings(document, pdf_path):
    doc = document.model_copy(deep=True)
    before = digest(semantic_content(doc))
    heading_refs = {x.self_ref for x in doc.texts if isinstance(x, (SectionHeaderItem, TitleItem))}
    changes = []
    with pymupdf.open(pdf_path) as pdf:
        toc_pages = set()
        index_pages = {
            p.page_no
            for table in doc.tables
            if table.label == DocItemLabel.DOCUMENT_INDEX
            for p in table.prov
        }
        for page in pdf:
            first = " ".join(page.get_text("text").split())[:500]
            if re.search(r"FORM\s+10-K\s+CROSS-REFERENCE\s+INDEX", first, re.I) or (
                page.number + 1 in index_pages and re.search(r"TABLE\s+OF\s+CONTENTS", first, re.I)
            ):
                toc_pages.add(page.number + 1)
        native = {}

        def style(item):
            if not item.prov:
                return None
            p = item.prov[0]
            page = pdf[p.page_no - 1]
            if p.page_no not in native:
                native[p.page_no] = [
                    s
                    for b in page.get_text("dict")["blocks"]
                    if "lines" in b
                    for line in b["lines"]
                    for s in line["spans"]
                ]
            box = p.bbox.to_top_left_origin(page.rect.height)
            rect = pymupdf.Rect(box.l, box.t, box.r, box.b)
            votes = Counter()
            for span in native[p.page_no]:
                sb = pymupdf.Rect(span["bbox"])
                if sb.get_area() and (sb & rect).get_area() / sb.get_area() > 0.65:
                    votes[(round(span["size"], 1), span["font"])] += len(span["text"])
            return votes.most_common(1)[0][0] if votes else None

        headings = [x for x in doc.texts if isinstance(x, SectionHeaderItem)]
        original_levels = {h.self_ref: h.level for h in headings}
        styles = {h.self_ref: style(h) for h in headings if h.prov}
        current_note = None
        for i, item in enumerate(doc.texts):
            if not isinstance(item, SectionHeaderItem):
                continue
            old = item.level
            reason = None
            page = item.prov[0].page_no if item.prov else None
            if page in toc_pages:
                values = item.model_dump(exclude={"level"})
                values["label"] = DocItemLabel.TEXT
                doc.texts[i] = TextItem.model_validate(values)
                changes.append({"ref": item.self_ref, "page": page, "reason": "toc_entry_not_section_start"})
                continue
            if EXHIBIT.fullmatch(item.text.strip()):
                item.level = 1
                current_note = None
                reason = "explicit_exhibit_boundary"
            elif match := NOTE.match(item.text.strip()):
                item.level = 2
                current_note = match.group(1)
                reason = "explicit_financial_note"
            elif (match := TABLE.match(item.text.strip())) and match.group(1) == current_note:
                item.level = 3
                reason = "table_belongs_to_matching_note"
            elif ABBREVIATION.match(item.text.strip()):
                key = styles.get(item.self_ref)
                peers = [
                    original_levels[h.self_ref]
                    for h in headings
                    if h.self_ref != item.self_ref
                    and h.prov
                    and key
                    and styles.get(h.self_ref) == key
                    and abs(h.prov[0].page_no - page) <= 10
                    and h.prov[0].page_no not in toc_pages
                    and not ABBREVIATION.match(h.text.strip())
                ]
                counts = Counter(peers)
                if len(peers) >= 2 and counts.most_common(1)[0][1] / len(peers) >= 0.8:
                    item.level = counts.most_common(1)[0][0]
                    reason = "abbreviation_uses_nearby_font_peers"
                else:
                    changes.append(
                        {
                            "ref": item.self_ref,
                            "page": page,
                            "reason": "abbreviation_unresolved",
                            "peer_levels": dict(counts),
                        }
                    )
            elif item.level <= 2:
                current_note = None
            if reason and old != item.level:
                changes.append(
                    {"ref": item.self_ref, "page": page, "reason": reason, "old": old, "new": item.level}
                )
    _linear_tree(doc, heading_refs)
    if digest(semantic_content(doc)) != before:
        raise RagError("Structural repair altered text, table content or coordinates.")
    return doc, {
        "contract": CONTRACT,
        "changes": changes,
        "toc_pages": sorted(toc_pages),
        "content_preserved": True,
        "complete_hierarchy_validation": False,
    }


def native_grid_candidate(table, pdf_path):
    """Only replace a matching, fully ruled, simple grid with no observed text loss.

    This produces a candidate and requires all previous cell words/numbers to be
    retained in the corresponding native cell. Unruled/spanning/ambiguous tables
    remain unchanged and can be routed to a different parser.
    """
    from docling_core.types.doc import BoundingBox, CoordOrigin

    from .text_checks import lexical

    report = {"ref": table.self_ref, "accepted": False}
    if len(table.prov) != 1 or not table.data.table_cells:
        return table, {**report, "reason": "requires_single_page_nonempty_grid"}
    cells = table.data.table_cells
    if len(cells) != table.data.num_rows * table.data.num_cols or any(
        not hasattr(c, "text") or c.row_span != 1 or c.col_span != 1 for c in cells
    ):
        return table, {**report, "reason": "spanning_cells_require_other_route"}
    with pymupdf.open(pdf_path) as pdf:
        page = pdf[table.prov[0].page_no - 1]
        box = table.prov[0].bbox.to_top_left_origin(page.rect.height)
        rect = pymupdf.Rect(box.l, box.t, box.r, box.b)
        matches = []
        for candidate in page.find_tables(strategy="lines_strict").tables:
            region = pymupdf.Rect(candidate.bbox)
            overlap = (region & rect).get_area()
            if (
                candidate.row_count == table.data.num_rows
                and candidate.col_count == table.data.num_cols
                and overlap / max(region.get_area(), rect.get_area()) > 0.9
            ):
                matches.append(candidate)
        if len(matches) != 1:
            return table, {**report, "reason": "no_unique_matching_ruled_grid"}
        candidate = matches[0]
        values = candidate.extract()
        updated = table.model_copy(deep=True)
        additions = []
        for cell in updated.data.table_cells:
            r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
            value = values[r][c] or ""
            old = lexical(cell.text)
            new = lexical(value)

            def numeric_atoms(text):
                return Counter(
                    s.replace(",", "") for s in re.findall(r"\(?[-−+]?\d[\d,]*(?:\.\d+)?%?\)?", text)
                )

            if old - new or numeric_atoms(cell.text) - numeric_atoms(value):
                return table, {
                    **report,
                    "reason": "native_cell_would_lose_existing_text",
                    "row": r,
                    "column": c,
                }
            # Retain cell text byte-for-byte when the native extraction adds no terms.
            if new - old:
                bounds = candidate.rows[r].cells[c]
                if bounds is None:
                    return table, {**report, "reason": "native_cell_geometry_missing"}
                additions.append({"row": r, "column": c, "added_terms": dict(new - old)})
                cell.text = value
                cell.bbox = BoundingBox(
                    l=bounds[0], t=bounds[1], r=bounds[2], b=bounds[3], coord_origin=CoordOrigin.TOPLEFT
                )
        if not additions:
            return table, {**report, "reason": "no_missing_content_recovered"}
        return updated, {
            **report,
            "accepted": True,
            "reason": "matching_grid_no_cell_word_loss",
            "additions": additions,
            "candidate_only": True,
        }
