import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ProvenanceItem,
    Size,
)

from financial_rag.common import RagError
from financial_rag.outline import (
    apply_item_reparents,
    apply_outline_patches,
    fingerprint_content,
    rebuild_scoped,
)


def sample():
    doc = DoclingDocument(name="scopes")
    for page in (1, 2, 3):
        doc.add_page(page_no=page, size=Size(width=600, height=800))

    def prov(page, text):
        return ProvenanceItem(
            page_no=page,
            charspan=(0, len(text)),
            bbox=BoundingBox(l=10, t=20, r=150, b=40, coord_origin=CoordOrigin.TOPLEFT),
        )

    a = doc.add_heading(text="Risk", level=1, prov=prov(1, "Risk"))
    b = doc.add_heading(text="Credit", level=1, prov=prov(2, "Credit"))
    c = doc.add_heading(text="Loans", level=2, prov=prov(2, "Loans"), parent=b)
    doc.add_text(label=DocItemLabel.TEXT, text="Loss (365)", prov=prov(2, "Loss (365)"), parent=c)
    d = doc.add_heading(text="Exhibit", level=2, prov=prov(3, "Exhibit"))
    return doc, a, b, c, d


def test_outline_patch_shifts_subtree_without_changing_source_or_crossing_attachment():
    doc, a, b, c, d = sample()
    before = fingerprint_content(doc)
    rebuild_scoped(doc, [1, 3])
    proposal = {"nest": [{"child_ref": b.self_ref, "parent_ref": a.self_ref, "evidence_pages": [1]}]}
    revised, report = apply_outline_patches(doc, proposal, [1, 3])
    assert report["content_unchanged"] and fingerprint_content(revised) == before
    assert revised.texts[1].parent.cref == a.self_ref
    assert revised.texts[2].level == 3
    assert revised.texts[-1].parent.resolve(revised).name == "source-range:3"
    count = len(revised.groups)
    rebuild_scoped(revised, [1, 3])
    assert len(revised.groups) == count
    assert fingerprint_content(revised) == before


def test_item_reparent_fixes_verified_layout_order_without_changing_content():
    doc, _, b, c, _ = sample()
    rebuild_scoped(doc, [1, 3])
    leaf = doc.texts[3]
    before = fingerprint_content(doc)
    revised, report = apply_item_reparents(
        doc,
        [{"child_ref": leaf.self_ref, "parent_ref": b.self_ref, "evidence_pages": [2]}],
        [1, 3],
    )
    assert report["content_unchanged"] and fingerprint_content(revised) == before
    assert revised.texts[3].parent.cref == b.self_ref
    assert revised.texts[2].parent.cref == b.self_ref


def test_item_reparent_allows_already_leveled_heading_but_rejects_level_change():
    doc, a, b, _, _ = sample()
    rebuild_scoped(doc, [1, 3])
    with pytest.raises(RagError):
        apply_item_reparents(
            doc,
            [{"child_ref": b.self_ref, "parent_ref": a.self_ref, "evidence_pages": [1]}],
            [1, 3],
        )


def test_rebuild_expands_a_structural_group_that_hides_a_heading():
    doc = DoclingDocument(name="hidden heading")
    doc.add_page(page_no=1, size=Size(width=600, height=800))

    def prov(text):
        return ProvenanceItem(
            page_no=1,
            charspan=(0, len(text)),
            bbox=BoundingBox(l=10, t=20, r=150, b=40, coord_origin=CoordOrigin.TOPLEFT),
        )

    first = doc.add_heading(text="Section 8", level=2, prov=prov("Section 8"))
    wrapper = doc.add_group(label=GroupLabel.FORM_AREA, name="layout wrapper", parent=doc.body)
    second = doc.add_heading(text="Section 9", level=2, prov=prov("Section 9"), parent=wrapper)
    paragraph = doc.add_text(
        label=DocItemLabel.TEXT, text="Rank text", prov=prov("Rank text"), parent=wrapper
    )
    rebuild_scoped(doc, [1])
    assert second.parent.resolve(doc).name == "source-range:1"
    assert paragraph.parent.cref == second.self_ref
    assert first.parent.resolve(doc).name == "source-range:1"
    assert wrapper.parent is None and wrapper.children == []


@pytest.mark.parametrize("child,parent,evidence", [(0, 1, [1]), (4, 0, [1]), (1, 0, [99])])
def test_outline_rejects_forward_parent_cross_scope_and_invalid_evidence(child, parent, evidence):
    doc, *_ = sample()
    before = doc.model_dump()
    with pytest.raises(RagError):
        apply_outline_patches(
            doc,
            {
                "nest": [
                    {
                        "child_ref": doc.texts[child].self_ref,
                        "parent_ref": doc.texts[parent].self_ref,
                        "evidence_pages": evidence,
                    }
                ]
            },
            [1, 3],
        )
    assert doc.model_dump() == before
