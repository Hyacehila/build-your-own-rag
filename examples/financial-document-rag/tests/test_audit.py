from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import pytest

from financial_rag.chunking import current_index
from financial_rag.common import RagError, sha256
from financial_rag.diagnostics.audit import audit_data, fragment_pages, missing_header_terms, page_node_text
from financial_rag.storage import Store


def test_audit_is_read_only_and_source_only(local, monkeypatch):
    config, writable = local
    from financial_rag.api import ModelAPI

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must never construct a model client")

    monkeypatch.setattr(ModelAPI, "__init__", forbidden)
    manifest = writable.meta("dataset")
    # Prove the audit does not require or read benchmark questions or answers.
    for name in ("queries.jsonl", "labels.jsonl"):
        (Path(manifest["prepared_dir"]) / name).unlink()
    before = sha256(writable.root / "experiment.sqlite")
    with Store(writable.root, read_only=True) as store:
        result = audit_data(config, store, writable.root / "audit", gallery=False)
        assert result["errors"] == 0
        assert result["api_calls_added"] == 0
        assert result["tables"]["numeric_token_loss_flags"] == 0
        assert result["tables"]["chunks_missing_declared_header_terms"] == 0
        assert result["indices"]["A"]["token_target"] == config.chunking.max_tokens
        assert store.db.total_changes == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.set_meta("not_allowed", True)
    assert sha256(writable.root / "experiment.sqlite") == before
    with pytest.raises(RagError, match="read_only"):
        audit_data(config, writable, writable.root / "bad-audit", gallery=False)


def test_audit_detects_altered_chunk_content_and_fts(local):
    config, store = local
    index = current_index(config, store, "flat")
    chunk = store.chunks(index["id"])[0]
    chunk["text"] = "Unexpected altered content 99999"
    with store.db:
        store.db.execute("UPDATE chunks SET payload=? WHERE id=?", (json.dumps(chunk), chunk["id"]))
    output = store.root / "audit"
    with Store(store.root, read_only=True) as ro:
        result = audit_data(config, ro, output, gallery=False)
    assert result["issues_by_type"]["flat_text_differs_from_source"] == 1
    assert result["issues_by_type"]["fts_chunk_text_consistency"] == 1


def test_page_text_uses_orig_spans_and_tableitem_shape():
    node = {
        "kind": "list_item",
        "text": "Revenue 10 continued 20",
        "raw": {
            "orig": "- Revenue 10 continued 20",
            "prov": [{"page_no": 1, "charspan": [0, 12]}, {"page_no": 2, "charspan": [13, 25]}],
        },
    }
    assert page_node_text(node, 1) == "- Revenue 10"
    assert page_node_text(node, 2) == "continued 20"
    toc = {
        "kind": "document_index",
        "text": "serialized table",
        "raw": {
            "data": {"table_cells": [{"text": "Note 12"}, {"text": "125"}]},
            "prov": [{"page_no": 1, "charspan": [0, 0]}],
        },
    }
    assert page_node_text(toc, 1) == "Note 12\n125"


def test_fragment_alignment_does_not_silently_guess():
    node = {
        "text": "First page. Second page.",
        "raw": {
            "orig": "First page. Second page.",
            "prov": [{"page_no": 1, "charspan": [0, 11]}, {"page_no": 2, "charspan": [12, 24]}],
        },
    }
    # The trailing span must be in bounds for any alignment to be accepted.
    node["raw"]["prov"][1]["charspan"][1] = len(node["raw"]["orig"])
    chunk = {
        "headings": ["Title"],
        "text": "Title\nSecond  page.",
        "sources": [{"page_number": 1}, {"page_number": 2}],
    }
    assert fragment_pages(chunk, [node]) == {
        "status": "aligned",
        "fragment_pages": [2],
        "extra_node_pages": [1],
    }
    altered = copy.deepcopy(node)
    altered["raw"]["prov"][1]["charspan"][1] += 2
    assert fragment_pages(chunk, [altered])["status"] == "invalid_original_charspan"
    assert fragment_pages(chunk, [node, node])["status"] == "not_uniquely_aligned"


def test_header_diagnostic_checks_all_header_rows_in_body():
    cells = [{"text": "Pretax", "column_header": True}, {"text": "2024", "column_header": True}]
    fragment = {"headings": ["Results 2024"], "text": "Results 2024\n| Pretax |\n|---|\n| 50 |"}
    assert missing_header_terms(cells, fragment) == ["2024"]
    fragment["text"] += "\n| 2024 |"
    assert missing_header_terms(cells, fragment) == []


def test_gallery_escapes_document_content_and_rejects_stale_reviews(tmp_path):
    from financial_rag.diagnostics.audit_gallery import write_gallery

    sample = {
        "doc_id": "d",
        "page_number": 1,
        "visual_review": "unreviewed",
        "text": "</script><script>attack()</script>",
    }
    data = {"index_ids": {"flat": "a", "structured": "b"}, "samples": [sample]}
    notes = {
        "index_ids": {"flat": "old", "structured": "old"},
        "pages": [{"doc_id": "d", "page_number": 1, "note": "reviewed"}],
    }
    (tmp_path / "reviews.json").write_text(json.dumps(notes), encoding="utf-8")
    write_gallery(tmp_path, data)
    assert data["samples"][0]["visual_review"] == "unreviewed"
    assert "</script><script>attack()" not in (tmp_path / "index.html").read_text(encoding="utf-8")
    notes["index_ids"] = data["index_ids"]
    (tmp_path / "reviews.json").write_text(json.dumps(notes), encoding="utf-8")
    write_gallery(tmp_path, data)
    assert data["samples"][0]["visual_review"] == "reviewed"
