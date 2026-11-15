import json
from pathlib import Path

import pytest

from financial_rag.common import RagError
from financial_rag.document.imported import import_documents
from financial_rag.indexing.chunking import chunk, current_index


def make_manifest(config, store, tmp_path):
    source = store.meta("dataset")["documents"][0]
    parse_id = store.meta(f"parse:structured:{source['id']}")
    parsed = store.get("parses", parse_id)
    doc = json.loads(Path(parsed["snapshot"]).read_text(encoding="utf-8"))
    headings = [item["self_ref"] for item in doc["texts"] if item["label"] in {"title", "section_header"}]
    outline = [{"ref": item["self_ref"], "text": item["text"], "level": item.get("level"),
                "parent": item["parent"]["cref"] if item.get("parent") else None,
                "pages": sorted({p["page_no"] for p in item["prov"]})}
               for item in doc["texts"] if item["label"] in {"title", "section_header"}]
    (tmp_path / "document.json").write_text(json.dumps(doc), encoding="utf-8")
    review = {"doc_id": source["doc_id"], "reviewer": "test", "final_tree_reviewed": True,
              "reviewed_heading_refs": headings, "reviewed_headings": outline,
              "confirmed_wrong_parent_remaining": [], "representative_pages_viewed": [1]}
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    manifest = {"documents": [{"doc_id": source["doc_id"], "pdf": source["pdf_path"],
                               "document": "document.json", "review": "review.json"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, path, manifest


def test_imported_document_can_build_structured_index(local, tmp_path):
    config, store = local
    source, path, _ = make_manifest(config, store, tmp_path)
    result = import_documents(config, store, path)
    assert result["completed"] == [source["doc_id"]]
    chunk(config, store, "structured")
    assert store.chunks(current_index(config, store, "structured")["id"])


def test_imported_facts_support_answer_and_evaluation(configured, tmp_path):
    from financial_rag.evaluation import evaluate
    from financial_rag.query.retrieval import embed_index
    from financial_rag.query.runner import run

    config, store, _ = configured
    _, path, _ = make_manifest(config, store, tmp_path)
    import_documents(config, store, path)
    chunk(config, store, "structured")
    embed_index(config, store, "structured")
    assert run(config, store, "imported", ["B"], limit=1)["failures"] == 0
    assert evaluate(config, store, "imported")["pending"] == 0


def test_import_preserves_relative_image_after_delivery_is_removed(local, tmp_path):
    from PIL import Image

    config, store = local
    source, path, _ = make_manifest(config, store, tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    image_path = assets / "picture.png"
    Image.new("RGB", (2, 2), color="red").save(image_path)
    document_path = tmp_path / "document.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["pictures"][0]["image"] = {
        "mimetype": "image/png", "dpi": 72, "size": {"width": 2, "height": 2},
        "uri": "assets/picture.png",
    }
    document_path.write_text(json.dumps(document), encoding="utf-8")
    import_documents(config, store, path)
    image_path.unlink()
    imported_id = store.meta(f"parse:structured:{source['id']}")
    snapshot = json.loads(Path(store.get("parses", imported_id)["snapshot"]).read_text(encoding="utf-8"))
    assert snapshot["pictures"][0]["image"]["uri"].startswith("data:image/png;base64,")


def test_import_rejects_missing_relative_image(local, tmp_path):
    config, store = local
    source, path, _ = make_manifest(config, store, tmp_path)
    before = store.meta(f"parse:structured:{source['id']}")
    document_path = tmp_path / "document.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["pictures"][0]["image"] = {
        "mimetype": "image/png", "dpi": 72, "size": {"width": 2, "height": 2},
        "uri": "assets/missing.png",
    }
    document_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RagError):
        import_documents(config, store, path)
    assert store.meta(f"parse:structured:{source['id']}") == before


@pytest.mark.parametrize("case", ["missing", "identity", "review", "invalid_page", "source_pdf", "outline", "review_page"])
def test_import_rejects_invalid_delivery_without_changing_active_parse(local, tmp_path, case):
    config, store = local
    source, path, manifest = make_manifest(config, store, tmp_path)
    prior = store.meta(f"parse:structured:{source['id']}")
    if case == "missing":
        manifest["documents"][0]["document"] = "missing.json"
    elif case == "identity":
        manifest["documents"][0]["doc_id"] = "incorrect"
    elif case == "review":
        (tmp_path / "review.json").write_text('{}', encoding="utf-8")
    elif case == "invalid_page":
        document_path = tmp_path / "document.json"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        document["pages"].pop("3")
        document_path.write_text(json.dumps(document), encoding="utf-8")
    elif case == "source_pdf":
        import pymupdf

        other = tmp_path / "other" / source["filename"]
        other.parent.mkdir()
        with pymupdf.open() as pdf:
            for _ in range(3):
                pdf.new_page()
            pdf.save(other)
        manifest["documents"][0]["pdf"] = str(other)
    elif case == "outline":
        review_path = tmp_path / "review.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["reviewed_headings"][0]["parent"] = "#/wrong"
        review_path.write_text(json.dumps(review), encoding="utf-8")
    else:
        review_path = tmp_path / "review.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["representative_pages_viewed"] = [4]
        review_path.write_text(json.dumps(review), encoding="utf-8")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RagError):
        import_documents(config, store, path)
    assert store.meta(f"parse:structured:{source['id']}") == prior
