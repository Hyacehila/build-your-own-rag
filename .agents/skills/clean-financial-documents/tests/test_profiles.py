import pytest
from clean_documents.cleaning_profiles import apply_profile
from clean_documents.common import RagError
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, GroupLabel, ListGroup, Size
from test_page_exclusions import provenance


def test_navigation_and_promoted_header_preserve_descendant_body():
    doc = DoclingDocument(name="review")
    for p in [1, 2]:
        doc.add_page(page_no=p, size=Size(width=600, height=800))
    nav = doc.add_heading(text="Contents", level=1, prov=provenance(1, "Contents"))
    body = doc.add_text(
        label=DocItemLabel.TEXT,
        text="Original facts 365",
        parent=nav,
        prov=provenance(1, "Original facts 365"),
    )
    header = doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="Notes",
        prov=provenance(1, "Notes"),
        content_layer=ContentLayer.FURNITURE,
    )
    profile = {
        "excluded_pages": [],
        "boundaries": [1],
        "heading_levels": {nav.self_ref: 1},
        "navigation_refs": [nav.self_ref],
        "promote": [{"ref": header.self_ref, "level": 1, "before_ref": body.self_ref, "evidence_pages": [1]}],
    }
    revised, report = apply_profile(doc, profile)
    assert report["content_unchanged"]
    assert revised.texts[0].content_layer == ContentLayer.FURNITURE
    assert revised.texts[0].parent.cref != "#/furniture"
    assert revised.texts[1].parent.cref == header.self_ref
    assert revised.texts[1].text == "Original facts 365"
    assert revised.texts[2].label == DocItemLabel.SECTION_HEADER
    assert doc.texts[2].label == DocItemLabel.PAGE_HEADER
    assert "Contents" not in revised.export_to_markdown()
    assert "Original facts 365" in revised.export_to_markdown()
    assert "Contents" in revised.export_to_markdown(included_content_layers=set(ContentLayer))
    assert revised.validate_tree(revised.body)
    revised._validate_rules()
    profile["promote"][0]["evidence_pages"] = [2]
    with pytest.raises(RagError, match="same-page"):
        apply_profile(doc, profile)


def test_navigation_list_keeps_its_subtype_and_source_text():
    doc = DoclingDocument(name="mixed navigation")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    group = doc.add_group(label=GroupLabel.LIST)
    item = doc.add_list_item(
        text="Contents entry 17",
        enumerated=True,
        marker="a.",
        prov=provenance(1, "Contents entry 17"),
        parent=group,
    )
    visible = doc.add_list_item(
        text="Actual disclosure",
        enumerated=True,
        marker="b.",
        prov=provenance(1, "Actual disclosure"),
        parent=group,
    )
    profile = {
        "excluded_pages": [],
        "boundaries": [1],
        "heading_levels": {},
        "navigation_refs": [item.self_ref],
    }
    revised, report = apply_profile(doc, profile)
    assert report["content_unchanged"] and revised.texts[0].marker == "a."
    assert revised.texts[0].text == item.text and revised.texts[0].content_layer == ContentLayer.FURNITURE
    assert isinstance(revised.texts[0].parent.resolve(revised), ListGroup)
    roundtrip = DoclingDocument.model_validate(revised.model_dump(mode="json"))
    assert roundtrip.validate_tree(roundtrip.body)
    roundtrip._validate_rules()
    assert "Contents entry 17" not in roundtrip.export_to_markdown()
    assert "Actual disclosure" in roundtrip.export_to_markdown()
    assert roundtrip.texts[1].self_ref == visible.self_ref
