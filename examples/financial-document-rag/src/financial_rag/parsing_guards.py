"""Opt-in guards around the pinned Docling reading-order stage.

Preserve the upstream parser as the default so existing experiment identities do
not change. This route rejects geometrically unsafe joins and locates merged
text against orig, including literal line-end hyphens.
"""

from __future__ import annotations

from collections import defaultdict

from docling.models.stages.reading_order.readingorder_model import ReadingOrderModel
from docling_core.types.doc import DocItemLabel, ProvenanceItem


def merge_rejection(left, right, elements):
    if right.page_no < left.page_no or right.page_no > left.page_no + 1:
        return "nonadjacent_or_reversed_pages"
    if left.page_no != right.page_no:
        return None
    # Reading-order elements use bottom-left coordinates. A full-width asset
    # partitions the columns into separate vertical zones on the same page.
    a, b = (left.t + left.b) / 2, (right.t + right.b) / 2
    for barrier in elements:
        if barrier.cid in (left.cid, right.cid) or barrier.page_no != left.page_no:
            continue
        if barrier.label not in (DocItemLabel.TABLE, DocItemLabel.PICTURE):
            continue
        if barrier.r - barrier.l < 0.7 * left.page_size.width:
            continue
        y = (barrier.t + barrier.b) / 2
        if min(a, b) < y < max(a, b):
            return "crosses_full_width_asset"
    return None


def order_page_zones(elements):
    """Keep the model's order inside zones delimited by full-width assets.

    Never guess around overlapping barriers or an element spanning a barrier.
    Such regions need a different layout parse, not an arbitrary sort by x/y.
    """
    pages = defaultdict(list)
    for element in elements:
        pages[element.page_no].append(element)
    result, changes = [], []
    for page_no, items in pages.items():
        barriers = sorted(
            (
                e
                for e in items
                if e.label in (DocItemLabel.TABLE, DocItemLabel.PICTURE)
                and e.r - e.l >= 0.8 * e.page_size.width
                and e.t - e.b < 0.8 * e.page_size.height
            ),
            key=lambda e: -e.t,
        )
        ids = {e.cid for e in barriers}
        overlaps = any(a.b < b.t for a, b in zip(barriers, barriers[1:]))
        straddles = any(e.cid not in ids and e.t > (b.t + b.b) / 2 > e.b for e in items for b in barriers)
        if not barriers or overlaps or straddles:
            result.extend(items)
            continue
        slots = {e.cid: 2 * i + 1 for i, e in enumerate(barriers)}

        def zone(e):
            return slots.get(e.cid, 2 * sum((e.t + e.b) / 2 < (b.t + b.b) / 2 for b in barriers))

        ordered = sorted(items, key=zone)
        if [e.cid for e in ordered] != [e.cid for e in items]:
            changes.append(
                {"page": page_no, "before": [e.cid for e in items], "after": [e.cid for e in ordered]}
            )
        result.extend(ordered)
    return result, changes


class _GuardedPredictor:
    def __init__(self, delegate, trace):
        self.delegate, self.trace = delegate, trace
        self.zone_changes = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def predict_reading_order(self, *, page_elements):
        proposed = self.delegate.predict_reading_order(page_elements=page_elements)
        ordered, self.zone_changes = order_page_zones(proposed)
        return ordered

    def predict_merges(self, *, sorted_elements):
        proposed = self.delegate.predict_merges(sorted_elements=sorted_elements)
        by_id = {e.cid: e for e in sorted_elements}
        result = {}
        for source, targets in proposed.items():
            kept = []
            previous = by_id[source]
            for target in targets:
                candidate = by_id[target]
                reason = merge_rejection(previous, candidate, sorted_elements)
                if reason:
                    self.trace.append(
                        {
                            "source_cid": source,
                            "target_cid": target,
                            "source_page": previous.page_no,
                            "target_page": candidate.page_no,
                            "reason": reason,
                        }
                    )
                    # The remainder of a proposed chain was based on the rejected
                    # predecessor; leave all remaining nodes independently readable.
                    break
                kept.append(target)
                previous = candidate
            if kept:
                result[source] = kept
        return result


class GuardedReadingOrderModel(ReadingOrderModel):
    def __init__(self, options):
        super().__init__(options)
        self.rejected_merges = []
        self.ro_model = _GuardedPredictor(self.ro_model, self.rejected_merges)
        self.page_heights = {}

    def __call__(self, conv_res):
        self.rejected_merges.clear()
        self.page_heights = {p.page_no: p.size.height for p in conv_res.pages if p.size}
        return super().__call__(conv_res)

    def _merge_elements(self, element, merged_elem, new_item, page_height):
        if type(element) is not type(merged_elem) or merged_elem.label != new_item.label:
            raise ValueError("Only matching element types and labels may be merged")
        separator = "" if new_item.text.endswith(("-", "\u00ad")) else " "
        start = len(new_item.orig) + len(separator)
        new_item.text += separator + merged_elem.text
        new_item.orig += separator + merged_elem.text
        new_item.prov.append(
            ProvenanceItem(
                page_no=merged_elem.page_no,
                charspan=(start, len(new_item.orig)),
                bbox=merged_elem.cluster.bbox.to_bottom_left_origin(
                    self.page_heights.get(merged_elem.page_no, page_height)
                ),
            )
        )
        if new_item.hyperlink != merged_elem.hyperlink:
            new_item.hyperlink = None
