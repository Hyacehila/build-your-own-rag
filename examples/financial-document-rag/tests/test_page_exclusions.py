import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

from financial_rag.common import RagError
from financial_rag.page_exclusions import exclude_pages
from financial_rag.parsing import docling_nodes


def provenance(page, text):
    return ProvenanceItem(
        page_no=page,
        charspan=(0, len(text)),
        bbox=BoundingBox(l=10, t=20, r=150, b=40, coord_origin=CoordOrigin.TOPLEFT),
    )


def test_exclude_navigation_keeps_retained_descendants_and_original_pages():
    document = DoclingDocument(name="navigation")
    for page in (1, 2, 3):
        document.add_page(page_no=page, size=Size(width=600, height=800))
    contents = document.add_heading(text="Part IV", level=1, prov=provenance(1, "Part IV"))
    real = document.add_heading(
        text="Risk Factors", level=2, prov=provenance(3, "Risk Factors"), parent=contents
    )
    document.add_text(
        label=DocItemLabel.TEXT, text="Loss (365)", parent=real, prov=provenance(3, "Loss (365)")
    )
    before = document.model_dump(mode="json")
    candidate, report = exclude_pages(document, {1})
    assert document.model_dump(mode="json") == before
    assert set(candidate.pages) == {1, 2, 3}
    assert [x.text for x in candidate.texts] == ["Risk Factors", "Loss (365)"]
    assert candidate.texts[0].parent.cref == "#/body"
    assert candidate.texts[1].parent.cref == candidate.texts[0].self_ref
    assert candidate.texts[1].prov[0].model_dump() == document.texts[2].prov[0].model_dump()
    nodes = docling_nodes(candidate, {"doc_id": "d", "id": "version", "pages": 3}, "candidate")
    assert all(s["page_index"] == 2 for n in nodes for s in n["sources"])
    assert report["retained_content_unchanged"]
    assert report["old_to_new_refs"][real.self_ref] == candidate.texts[0].self_ref


def test_exclude_pages_refuses_mixed_page_nodes_and_invalid_page_labels():
    document = DoclingDocument(name="cross-page")
    for page in (1, 2):
        document.add_page(page_no=page, size=Size(width=600, height=800))
    node = document.add_text(
        label=DocItemLabel.TEXT, text="navigation body", prov=provenance(1, "navigation")
    )
    node.prov.append(provenance(2, "body"))
    with pytest.raises(RagError, match="spans excluded and retained"):
        exclude_pages(document, {1})
    for pages in ({0}, {3}, {True}, set()):
        with pytest.raises(RagError, match="physical PDF"):
            exclude_pages(document, pages)
