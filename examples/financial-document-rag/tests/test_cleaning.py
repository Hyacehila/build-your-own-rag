import sys
from pathlib import Path

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, GroupLabel, ListGroup, Size
from test_page_exclusions import provenance

from financial_rag.asset_roles import empty_tables_as_pictures
from financial_rag.cleaning import build, export_release, restore_release, risk_pages
from financial_rag.cleaning_profiles import apply_profile
from financial_rag.common import RagError, read_json, write_json
from financial_rag.config import Config, Tokens
from financial_rag.final_outline import accept, heading_rows, outline
from financial_rag.outline import apply_item_reparents, apply_outline_patches
from financial_rag.parsing import docling_nodes, load_docling
from financial_rag.storage import Store
from financial_rag.tokenization import Tokenizer


def final_outline_receipt(root, release):
    """Create a deliberately source-bound reviewer fixture, never a production receipt."""
    row = release["documents"][0]
    with Store(root, read_only=True) as store:
        data = outline(store, row)
    first = data["headings"][0]
    return data, {
        **{key: data[key] for key in ("contract", "doc_id", "parse_id", "snapshot_sha256", "outline_sha256")},
        "reviewer": "pytest fixture reviewer",
        "final_tree_reviewed": True,
        "reviewed_heading_refs": [node["ref"] for node in data["headings"]],
        "contents_pages_viewed": [1],
        "representative_pages_viewed": [1],
        "suspicious_pages_viewed": [],
        "confirmed_wrong_parent_remaining": [],
        "uncertain_exceptions": [],
        "expected_relations": [
            {
                "child_ref": first["ref"],
                "relation": "parent_heading_ref",
                "expected_parent_ref": first["parent_heading_ref"],
                "evidence_pages": [first["pages"][0]],
            }
        ],
    }


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


def test_furniture_reclassification_retains_body_descendants_and_reading_order(tmp_path):
    from financial_rag.chunking import hybrid_chunks

    doc = DoclingDocument(name="navigation with body")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    nav = doc.add_text(label=DocItemLabel.TEXT, text="Navigation", prov=provenance(1, "Navigation"))
    body = doc.add_text(
        label=DocItemLabel.TEXT, text="Retained facts", parent=nav, prov=provenance(1, "Retained facts")
    )
    profile = {"excluded_pages": [], "boundaries": [1], "heading_levels": {}, "furniture": [nav.self_ref]}
    revised, report = apply_profile(doc, profile)
    assert report["content_unchanged"]
    assert revised.texts[0].content_layer == ContentLayer.FURNITURE
    assert revised.texts[1].parent.cref == nav.self_ref
    assert "Navigation" not in revised.export_to_markdown()
    assert "Retained facts" in revised.export_to_markdown()
    nodes = docling_nodes(revised, {"doc_id": "d", "id": "v", "pages": 1}, "p")
    by_ref = {n["ref"]: n for n in nodes}
    assert "#/furniture" not in by_ref
    assert by_ref[nav.self_ref]["reading_order"] is None
    assert by_ref[body.self_ref]["reading_order"] is not None
    snapshot = tmp_path / "document.json"
    write_json(snapshot, revised.model_dump(mode="json"))
    config = Config(work_dir=tmp_path, tokenizer=Tokens(kind="approx", estimated=True))
    chunks = hybrid_chunks({"snapshot": str(snapshot)}, nodes, config, Tokenizer(config.tokenizer))
    assert any("Retained facts" in chunk["text"] for chunk in chunks)
    assert all("Navigation" not in chunk["text"] for chunk in chunks)


def test_furniture_heading_does_not_change_body_heading_scope(tmp_path):
    from financial_rag.chunking import hybrid_chunks

    doc = DoclingDocument(name="layered headings")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    main = doc.add_heading(text="Main section", level=1, prov=provenance(1, "Main section"))
    hidden = doc.add_heading(
        text="Page heading", level=1, prov=provenance(1, "Page heading"), content_layer=ContentLayer.FURNITURE
    )
    body = doc.add_text(
        label=DocItemLabel.TEXT, text="Financial facts", prov=provenance(1, "Financial facts")
    )
    profile = {"excluded_pages": [], "boundaries": [1], "heading_levels": {main.self_ref: 1}}
    revised, _ = apply_profile(doc, profile)
    assert revised.texts[2].parent.cref == main.self_ref
    nodes = docling_nodes(revised, {"doc_id": "d", "id": "v", "pages": 1}, "p")
    by_ref = {n["ref"]: n for n in nodes}
    assert by_ref[hidden.self_ref]["reading_order"] is None
    assert by_ref[body.self_ref]["headings"] == ["Main section"]
    snapshot = tmp_path / "document.json"
    write_json(snapshot, revised.model_dump(mode="json"))
    config = Config(work_dir=tmp_path, tokenizer=Tokens(kind="approx", estimated=True))
    chunks = hybrid_chunks({"snapshot": str(snapshot)}, nodes, config, Tokenizer(config.tokenizer))
    assert any("Financial facts" in chunk["text"] for chunk in chunks)
    assert all("Page heading" not in chunk["text"] for chunk in chunks)


@pytest.mark.parametrize(
    "layer", [ContentLayer.FURNITURE, ContentLayer.BACKGROUND, ContentLayer.INVISIBLE, ContentLayer.NOTES]
)
def test_non_body_heading_is_not_a_reading_position_or_body_section(layer):
    doc = DoclingDocument(name="non-body content")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    heading = doc.add_heading(
        text="Hidden heading", level=1, content_layer=layer, prov=provenance(1, "Hidden heading")
    )
    section = doc.add_heading(
        text="Visible section", level=2, parent=heading, prov=provenance(1, "Visible section")
    )
    child = doc.add_text(
        label=DocItemLabel.TEXT, text="Visible child", parent=section, prov=provenance(1, "Visible child")
    )
    nodes = {n["ref"]: n for n in docling_nodes(doc, {"doc_id": "d", "id": "v", "pages": 1}, "p")}
    assert nodes[heading.self_ref]["reading_order"] is None
    assert nodes[child.self_ref]["reading_order"] is not None
    assert nodes[child.self_ref]["headings"] == ["Visible section"]
    headings = heading_rows(list(nodes.values()))
    assert [item["ref"] for item in headings] == [section.self_ref]
    assert headings[0]["parent_heading_ref"] is None


def test_outline_edits_reject_a_non_body_parent():
    doc = DoclingDocument(name="layered outline")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    hidden = doc.add_heading(
        text="Page heading", level=1, content_layer=ContentLayer.FURNITURE, prov=provenance(1, "Page heading")
    )
    section = doc.add_heading(text="Main section", level=2, prov=provenance(1, "Main section"))
    body = doc.add_text(label=DocItemLabel.TEXT, text="Retained facts", prov=provenance(1, "Retained facts"))
    move = {"child_ref": body.self_ref, "parent_ref": hidden.self_ref, "evidence_pages": [1]}
    with pytest.raises(RagError, match="heading parent"):
        apply_item_reparents(doc, [move], [1])
    patch = {"child_ref": section.self_ref, "parent_ref": hidden.self_ref, "evidence_pages": [1]}
    with pytest.raises(RagError, match="existing, preceding headings"):
        apply_outline_patches(doc, {"nest": [patch]}, [1])


@pytest.mark.parametrize("excluded_pages", [[], [{"page_number": 1}]])
def test_cleaning_rejects_legacy_furniture_without_changing_source(excluded_pages):
    doc = DoclingDocument(name="legacy furniture")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    header = doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="Old header",
        content_layer=ContentLayer.FURNITURE,
        prov=provenance(1, "Old header"),
    )
    legacy = doc.__dict__["furniture"]
    doc.body.children = [r for r in doc.body.children if r.cref != header.self_ref]
    header.parent = legacy.get_ref()
    legacy.children.append(header.get_ref())
    original = doc.model_dump(mode="json")
    with pytest.raises(RagError, match="raw parse with content layers"):
        apply_profile(doc, {"excluded_pages": excluded_pages, "boundaries": [1], "heading_levels": {}})
    assert doc.model_dump(mode="json") == original


def test_default_cli_paths_keep_new_facts_and_reviews_separate(monkeypatch):
    from financial_rag import cleaning, config, final_outline
    from financial_rag.diagnostics import cleaning_audit

    paths = {}

    def build_candidate(workspace, profiles, output, settings):
        paths["build"] = output
        return {}

    def export_candidate(output, destination):
        paths["export"] = (output, destination)
        return {}

    def make_packets(root, reviews, packet_size):
        paths["packets"] = (root, reviews)
        return {}

    def audit(root):
        paths["audit"] = root
        return []

    monkeypatch.setattr(config, "load_config", lambda _: None)
    monkeypatch.setattr(cleaning, "build", build_candidate)
    monkeypatch.setattr(cleaning, "export_release", export_candidate)
    monkeypatch.setattr(final_outline, "packets", make_packets)
    monkeypatch.setattr(cleaning_audit, "audit_pages", audit)
    monkeypatch.setattr(sys, "argv", ["cleaning", "build"])
    cleaning.main()
    monkeypatch.setattr(sys, "argv", ["cleaning", "export"])
    cleaning.main()
    monkeypatch.setattr(sys, "argv", ["final_outline", "packets"])
    final_outline.main()
    cleaning_audit.main([])

    output = Path(".local/facts-v3-release")
    assert paths == {
        "build": output,
        "export": (output, Path("handoff/finance-facts-v3.sqlite")),
        "packets": (output, Path(".local/final-outline-v3")),
        "audit": output,
    }


def test_fact_release_roundtrip_and_version_guard(local, tmp_path, monkeypatch):
    config, source = local
    doc = source.meta("dataset")["documents"][0]
    raw = source.get("parses", source.meta("parse:structured:" + doc["id"]))
    workspace = tmp_path / "prepare"
    profiles = tmp_path / "profiles"
    snapshot = load_docling(raw)
    profile = {
        "doc_id": doc["doc_id"],
        "pdf_sha256": doc["sha256"],
        "raw_snapshot_sha256": raw["snapshot_sha256"],
        "excluded_pages": [],
        "boundaries": [1],
        "heading_levels": {
            h.self_ref: h.level for h in snapshot.texts if h.label == DocItemLabel.SECTION_HEADER
        },
        "reviewed_pages": [1, 2, 3],
        "unresolved": [],
    }
    write_json(workspace / "sources.json", [{"document": doc, "raw": raw}])
    write_json(workspace / "diagnostics" / (doc["doc_id"] + ".json"), {"issues": []})
    write_json(profiles / (doc["doc_id"] + ".json"), profile)
    output = tmp_path / "release"
    result = build(workspace, profiles, output, config)
    assert result["foreign_key_errors"] == 0
    database = tmp_path / "facts.sqlite"
    with pytest.raises(RagError, match="semantic acceptance"):
        export_release(output, database)
    _, receipt = final_outline_receipt(output, result)
    review_dir = tmp_path / "reviews" / doc["doc_id"]
    write_json(review_dir / "receipt.json", receipt)
    accepted = accept(output, tmp_path / "reviews")
    assert accepted["documents"][0]["headings_reviewed"] == len(receipt["reviewed_heading_refs"])
    export_release(output, database)
    restored = restore_release(database, tmp_path / "restored", output / "raw")
    assert restored["release_id"] == result["release_id"]
    from financial_rag import cleaning

    with monkeypatch.context() as temporary:
        temporary.setattr(cleaning, "CONTRACT", "reviewed-financial-facts-v2")
        with pytest.raises(RagError, match="inputs changed"):
            build(workspace, profiles, output, config)
    assert read_json(output / "release.json")["release_id"] == result["release_id"]
    assert (tmp_path / "restored/release.json").exists()
    with pytest.raises(RagError, match="already exists"):
        export_release(output, database)
    profile["reviewed_pages"] = [1]
    write_json(profiles / (doc["doc_id"] + ".json"), profile)
    with pytest.raises(RagError, match="inputs changed"):
        build(workspace, profiles, output, config)
    assert read_json(output / "release.json")["release_id"] == result["release_id"]


def test_final_outline_acceptance_rejects_incomplete_tree_coverage(local, tmp_path):
    config, source = local
    doc = source.meta("dataset")["documents"][0]
    raw = source.get("parses", source.meta("parse:structured:" + doc["id"]))
    snapshot = load_docling(raw)
    workspace, profiles, output = tmp_path / "prepare", tmp_path / "profiles", tmp_path / "release"
    write_json(workspace / "sources.json", [{"document": doc, "raw": raw}])
    write_json(workspace / "diagnostics" / (doc["doc_id"] + ".json"), {"issues": []})
    write_json(
        profiles / (doc["doc_id"] + ".json"),
        {
            "doc_id": doc["doc_id"],
            "pdf_sha256": doc["sha256"],
            "raw_snapshot_sha256": raw["snapshot_sha256"],
            "excluded_pages": [],
            "boundaries": [1],
            "heading_levels": {
                h.self_ref: h.level for h in snapshot.texts if h.label == DocItemLabel.SECTION_HEADER
            },
            "reviewed_pages": [1],
            "unresolved": [],
        },
    )
    result = build(workspace, profiles, output, config)
    _, receipt = final_outline_receipt(output, result)
    receipt["reviewed_heading_refs"] = []
    write_json(tmp_path / "reviews" / doc["doc_id"] / "receipt.json", receipt)
    with pytest.raises(RagError, match="coverage"):
        accept(output, tmp_path / "reviews")
    with Store(output, read_only=True) as store:
        assert store.meta("outline_acceptance") is None


def test_risk_routing_retains_physical_pages_and_excludes_navigation():
    source = {"document": {"pages": 10}}
    diagnostic = {
        "issues": [
            {"check": "table_region_numbers_not_assigned", "pages": [4, 9]},
            {"check": "table_without_declared_header", "pages": [1]},
        ]
    }
    core, pages = risk_pages(source, diagnostic, {"excluded_pages": [{"page_number": 3}]})
    assert core == [4, 9] and pages == [4, 5, 8, 9, 10]


def test_empty_diagram_reclassification_remaps_retained_table_and_caption():
    from docling_core.types.doc import TableCell, TableData

    from financial_rag.parsing import docling_nodes

    doc = DoclingDocument(name="assets")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    empty = doc.add_table(data=TableData(num_rows=0, num_cols=0, table_cells=[]), prov=provenance(1, ""))
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Diagram", prov=provenance(1, "Diagram"))
    empty.captions.append(caption.get_ref())
    real = doc.add_table(
        data=TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    text="365",
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                )
            ],
        ),
        prov=provenance(1, "365"),
    )
    result, mapping, decisions = empty_tables_as_pictures(doc, [empty.self_ref])
    assert len(result.tables) == 1 and result.tables[0].data.table_cells[0].text == "365"
    assert len(result.pictures) == 1 and result.pictures[0].captions[0].cref == caption.self_ref
    assert mapping[real.self_ref] == "#/tables/0" and decisions[0]["original"]["data"]["table_cells"] == []
    assert result.pictures[0].prov == empty.prov
    docling_nodes(result, {"doc_id": "assets", "id": "version", "pages": 1}, "clean")
    with pytest.raises(RagError, match="Only existing empty"):
        empty_tables_as_pictures(doc, [real.self_ref])
