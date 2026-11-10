"""The published teaching objects must match the real local data pipeline."""

import importlib.util
import json
import socket
from pathlib import Path

import pymupdf
import pytest

from financial_rag.common import RagError

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teaching_example", PROJECT / "tools/teaching_example.py")
TEACHING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEACHING)


def test_teaching_pipeline_matches_published_structure_and_reads_original_table(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Teaching data preparation must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    output = tmp_path / "teaching"
    result = TEACHING.build(output)
    document = json.loads((output / "document.json").read_text(encoding="utf-8"))
    reference = json.loads((PROJECT / "reference/document.json").read_text(encoding="utf-8"))
    assert document == reference
    assert (output / "structure.txt").read_text(encoding="utf-8") == (
        PROJECT / "reference/structure.txt"
    ).read_text(encoding="utf-8")
    with pymupdf.open(result["pdf"]) as pdf:
        assert len(pdf) == 3 and "100" in pdf[1].get_text()
    table = document["tables"][0]
    assert table["parent"]["$ref"] == "#/texts/3" and table["prov"][0]["page_no"] == 2
    assert document["texts"][3]["parent"]["$ref"] == "#/texts/0"
    assert result["chunk_counts"]["flat"] > 0 and result["chunk_counts"]["structured"] > 0
    chunks = json.loads((output / "chunks.json").read_text(encoding="utf-8"))
    assert any("#/tables/0" in chunk["refs"] for chunk in chunks if chunk["index"] == "structured")
    readback = json.loads((output / "readback.json").read_text(encoding="utf-8"))
    evidence = readback["evidence"]
    assert readback["ref"] == "#/tables/0" and all(not item.get("truncated") for item in evidence)
    assert all(source["page_number"] == 2 for item in evidence for source in item["sources"])
    assert "Retail" in evidence[0]["text"] and "100" in evidence[0]["text"]


def test_teaching_output_does_not_overwrite_existing_material(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "note.txt"
    marker.write_text("keep this material", encoding="utf-8")
    with pytest.raises(RagError, match="Output already exists"):
        TEACHING.build(output)
    assert marker.read_text(encoding="utf-8") == "keep this material"
    assert not (output / "experiment.sqlite").exists()
