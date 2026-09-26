import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, Size
from test_page_exclusions import provenance

from financial_rag.asset_roles import empty_tables_as_pictures
from financial_rag.cleaning import build, export_release, restore_release, risk_pages
from financial_rag.cleaning_profiles import apply_profile
from financial_rag.common import RagError, read_json, write_json
from financial_rag.final_outline import accept, outline
from financial_rag.parsing import load_docling
from financial_rag.storage import Store


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
    assert revised.texts[0].parent.cref == "#/furniture"
    assert revised.texts[1].parent.cref == header.self_ref
    assert revised.texts[1].text == "Original facts 365"
    assert revised.texts[2].label == DocItemLabel.SECTION_HEADER
    assert doc.texts[2].label == DocItemLabel.PAGE_HEADER
    profile["promote"][0]["evidence_pages"] = [2]
    with pytest.raises(RagError, match="same-page"):
        apply_profile(doc, profile)


def test_navigation_list_keeps_its_subtype_and_source_text():
    doc = DoclingDocument(name="mixed navigation")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    item = doc.add_list_item(
        text="Contents entry 17", enumerated=True, marker="a.", prov=provenance(1, "Contents entry 17")
    )
    profile = {
        "excluded_pages": [],
        "boundaries": [1],
        "heading_levels": {},
        "navigation_refs": [item.self_ref],
    }
    revised, report = apply_profile(doc, profile)
    assert report["content_unchanged"] and revised.texts[0].marker == "a."
    assert revised.texts[0].text == item.text and revised.texts[0].parent.cref == "#/furniture"
    DoclingDocument.model_validate(revised.model_dump(mode="json"))


def test_fact_release_roundtrip_and_version_guard(local, tmp_path):
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
