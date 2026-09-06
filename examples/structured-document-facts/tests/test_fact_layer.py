"""Behavior checks against an original fixture, with network access disabled."""

import json
import socket
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, RefItem

from fact_layer import (
    FactError, Locator, build_bundle, convert_markdown, json_bytes,
    load_bundle, read_node, sha256, walk_body,
)

EXAMPLE = Path(__file__).resolve().parents[1]
SOURCE = EXAMPLE / "fixtures" / "product-review.md"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("The example must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "snapshot"
    build_bundle(SOURCE, output, "product-review")
    return output


def locator_for(bundle, ref):
    manifest, _ = load_bundle(bundle)
    return Locator(document_id=manifest.document_id, document_sha256=manifest.document_sha256, self_ref=ref)


def test_roundtrip_and_original_source(bundle):
    manifest, document = load_bundle(bundle)
    assert (bundle / "source.md").read_bytes() == SOURCE.read_bytes()
    assert manifest.source_sha256 == sha256(SOURCE.read_bytes())
    assert DoclingDocument.model_validate_json(
        json_bytes(document.model_dump(mode="json", by_alias=True))
    ) == document
    assert manifest.parser_versions == {"docling-slim": "2.120.1", "docling-core": "2.91.0"}
    assert document.pages == {}
    assert all(not item.prov for item in document.texts)


def test_heading_adaptation_is_explicit_and_preserves_content(bundle):
    _, document = load_bundle(bundle)
    parsed = DoclingDocument.model_validate_json((bundle / "parsed-document.json").read_bytes())
    # This pinned Markdown backend leaves the table at the root before adaptation.
    assert parsed.tables[0].parent.cref == "#/body"
    assert [item.text for item in parsed.texts] == [item.text for item in document.texts]
    assert [item.self_ref for item in parsed.texts] == [item.self_ref for item in document.texts]
    assert [item.self_ref for item, _ in walk_body(parsed)] == [item.self_ref for item, _ in walk_body(document)]
    results = [item for item in document.texts if item.text == "观察结果"]
    assert len(results) == 2
    paths = [read_node(bundle, locator_for(bundle, item.self_ref))["ancestors"] for item in results]
    assert paths[0][-1]["text"] == "新团队邀请实验"
    assert paths[1][-1]["text"] == "现有团队权限检查"


def test_table_and_list_relationships(bundle):
    _, document = load_bundle(bundle)
    table = document.tables[0]
    assert (table.data.num_rows, table.data.num_cols) == (3, 4)
    cells = {(cell.start_row_offset_idx, cell.start_col_offset_idx): cell for cell in table.data.table_cells}
    assert cells[0, 2].text == "调整后" and cells[0, 2].column_header
    assert cells[1, 2].text == "55%"
    assert cells[1, 3].text == "新团队在 7 天内完成首次邀请"
    read = read_node(bundle, locator_for(bundle, table.self_ref))
    assert [parent["text"] for parent in read["ancestors"]][-2:] == ["新团队邀请实验", "观察结果"]
    groups = [group for group in document.groups if group.label == "list"]
    assert [len(group.children) for group in groups] == [2, 2]
    assert all(ref.resolve(document).label == DocItemLabel.LIST_ITEM for group in groups for ref in group.children)


def test_image_uri_and_caption_without_fetching(bundle):
    _, document = load_bundle(bundle)
    picture = document.pictures[0]
    assert str(picture.image.uri) == "https://example.invalid/invitation-flow.png"
    assert picture.image.size.width == picture.image.size.height == 0
    assert len(picture.captions) == 1
    caption = picture.captions[0].resolve(document)
    assert caption.label == DocItemLabel.CAPTION
    assert caption.text == "图 1：虚构的新团队邀请路径，创建工作区后进入邀请页面。"
    assert len([item for item in document.texts if item.text == caption.text]) == 1


@pytest.mark.parametrize("changes", [
    {"document_id": "other-document"},
    {"document_sha256": "0" * 64},
])
def test_stale_locator_is_rejected(bundle, changes):
    data = locator_for(bundle, "#/tables/0").model_dump() | changes
    with pytest.raises(FactError, match="STALE_LOCATOR"):
        read_node(bundle, Locator(**data))


def test_missing_node_is_rejected(bundle):
    with pytest.raises(FactError, match="MISSING_REF"):
        read_node(bundle, locator_for(bundle, "#/texts/99999"))


@pytest.mark.parametrize("filename,error", [
    ("source.md", "SOURCE_MISMATCH"),
    ("document.json", "DOCUMENT_MISMATCH"),
])
def test_modified_snapshot_is_rejected(bundle, filename, error):
    path = bundle / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FactError, match=error):
        load_bundle(bundle)


def test_invalid_json_is_rejected_even_with_updated_hash(bundle):
    (bundle / "document.json").write_bytes(b"{not json}")
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    manifest["document_sha256"] = sha256(b"{not json}")
    (bundle / "manifest.json").write_bytes(json_bytes(manifest))
    with pytest.raises(FactError, match="INVALID_BUNDLE"):
        load_bundle(bundle)


def test_missing_manifest_is_an_incomplete_bundle(bundle):
    (bundle / "manifest.json").unlink()
    with pytest.raises(FactError, match="INVALID_BUNDLE"):
        load_bundle(bundle)


def test_broken_parent_and_cycle_are_rejected(bundle):
    _, document = load_bundle(bundle)
    document.texts[5].parent = RefItem(cref="#/body")
    with pytest.raises(FactError, match="parent mismatch"):
        list(walk_body(document))
    _, document = load_bundle(bundle)
    document.body.parent = document.body.get_ref()
    document.body.children.append(document.body.get_ref())
    with pytest.raises(FactError, match="repeated node"):
        list(walk_body(document))


def test_existing_snapshot_is_never_overwritten(bundle):
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    with pytest.raises(FactError, match="OUTPUT_EXISTS"):
        build_bundle(SOURCE, bundle, "product-review")
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir()}


def test_same_source_produces_same_document_bytes(bundle, tmp_path):
    other = tmp_path / "repeat"
    build_bundle(SOURCE, other, "product-review")
    for name in ("source.md", "parsed-document.json", "document.json", "manifest.json", "locator.json"):
        assert (bundle / name).read_bytes() == (other / name).read_bytes()


def test_published_reference_matches_real_conversion(bundle):
    for name in ("parsed-document.json", "document.json", "structure.txt"):
        assert (EXAMPLE / "reference" / name).read_bytes() == (bundle / name).read_bytes()


@pytest.mark.parametrize("source,name", [(b"", "empty.md"), (b"\xff", "bad.md"), (b"hello", "wrong.pdf")])
def test_invalid_sources(source, name):
    with pytest.raises(FactError, match="INVALID_SOURCE"):
        convert_markdown(source, name)
