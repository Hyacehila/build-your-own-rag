"""Conservative page-local table replacement and native-source span repair candidates."""

import re
from collections import defaultdict
from difflib import SequenceMatcher

import pymupdf

from .diagnostics.text_checks import numbers, recovery


def overlap(a, b):
    area = max(0, min(a.r, b.r) - max(a.l, b.l)) * max(0, min(a.b, b.b) - max(a.t, b.t))
    aa = (a.r - a.l) * (a.b - a.t)
    bb = (b.r - b.l) * (b.b - b.t)
    return area / max(aa + bb - area, 1e-9)


def repair_tables(document, cloud_document, suspect_refs, pdf_path):
    doc = document.model_copy(deep=True)
    by_page = defaultdict(list)
    for table in cloud_document.tables:
        if len(table.prov) == 1:
            by_page[table.prov[0].page_no].append(table)
    changes = []
    used = set()
    with pymupdf.open(pdf_path) as pdf:
        for i, table in enumerate(doc.tables):
            if table.self_ref not in suspect_refs:
                continue
            if len(table.prov) != 1:
                changes.append({"ref": table.self_ref, "accepted": False, "reason": "cross_page_source"})
                continue
            number = table.prov[0].page_no
            height = pdf[number - 1].rect.height
            a = table.prov[0].bbox.to_top_left_origin(height)
            candidates = sorted(
                [(overlap(a, t.prov[0].bbox.to_top_left_origin(height)), t) for t in by_page[number]],
                key=lambda x: -x[0],
            )
            if not candidates or candidates[0][0] < 0.6 or (len(candidates) > 1 and candidates[1][0] > 0.4):
                changes.append(
                    {
                        "ref": table.self_ref,
                        "page": number,
                        "accepted": False,
                        "reason": "no_unique_geometric_match",
                    }
                )
                continue
            score, new = candidates[0]
            if new.self_ref in used:
                changes.append(
                    {
                        "ref": table.self_ref,
                        "page": number,
                        "accepted": False,
                        "reason": "cloud_table_already_used",
                    }
                )
                continue
            b = new.prov[0].bbox.to_top_left_origin(height)
            region = pymupdf.Rect(min(a.l, b.l), min(a.t, b.t), max(a.r, b.r), max(a.b, b.b))
            native = pdf[number - 1].get_text("text", clip=region, sort=True)
            expected = numbers(native)
            before = numbers("\n".join(c.text for c in table.data.table_cells))
            after = numbers("\n".join(c.text for c in new.data.table_cells))
            old_recovery = recovery(expected, before)
            new_recovery = recovery(expected, after)
            accepted = (
                old_recovery is not None and new_recovery is not None and new_recovery >= old_recovery - 0.01
            )
            row = {
                "ref": table.self_ref,
                "cloud_ref": new.self_ref,
                "page": number,
                "iou": score,
                "accepted": accepted,
                "old_shape": [table.data.num_rows, table.data.num_cols],
                "new_shape": [new.data.num_rows, new.data.num_cols],
                "old_number_recovery": old_recovery,
                "new_number_recovery": new_recovery,
                "validation": "geometry and native numeric coverage; not a proof of every cell meaning",
            }
            if accepted:
                replacement = table.model_copy(deep=True)
                replacement.data = new.data.model_copy(deep=True)
                replacement.prov = [p.model_copy(deep=True) for p in new.prov]
                # Original captions, notes and stable node references remain attached.
                doc.tables[i] = replacement
                used.add(new.self_ref)
            changes.append(row)
    return doc, changes


def repair_invalid_spans(document, pdf_path):
    doc = document.model_copy(deep=True)
    changes = []
    with pymupdf.open(pdf_path) as pdf:
        for node in doc.texts:
            original = node.orig
            if not any(not 0 <= p.charspan[0] <= p.charspan[1] <= len(original) for p in node.prov):
                continue
            pieces = []
            for p in node.prov:
                page = pdf[p.page_no - 1]
                b = p.bbox.to_top_left_origin(page.rect.height)
                pieces.append(
                    page.get_text(
                        "text", clip=pymupdf.Rect(b.l - 0.5, b.t - 0.5, b.r + 0.5, b.b + 0.5), sort=True
                    ).strip()
                )
            restored = "\n".join(pieces)

            def normalize(s):
                return re.sub(r"\W+", "", s).casefold()

            similarity = SequenceMatcher(
                None, normalize(original), normalize(restored), autojunk=False
            ).ratio()
            accepted = similarity >= 0.98 and numbers(original) == numbers(restored)
            row = {
                "ref": node.self_ref,
                "pages": [p.page_no for p in node.prov],
                "similarity": similarity,
                "accepted": accepted,
            }
            if accepted:
                row.update(old_orig=original, new_orig=restored)
                node.orig = restored
                node.text = restored
                cursor = 0
                for p, piece in zip(node.prov, pieces):
                    p.charspan = (cursor, cursor + len(piece))
                    cursor += len(piece) + 1
            changes.append(row)
    return doc, changes
