"""Opt-in test of the learned local parser, with no paid API and no fake snapshot."""

import os

import pytest

from financial_rag.chunking import chunk
from financial_rag.config import Config, Parsing, Tokens
from financial_rag.fixture import create_fixture
from financial_rag.parsing import active_parses, parse
from financial_rag.storage import Store


@pytest.mark.docling
@pytest.mark.skipif(
    not os.environ.get("FINANCE_DOCLING_ARTIFACTS"), reason="Local Docling weights not provided"
)
def test_real_pdf_parser_heading_hierarchy_tables_and_sources(tmp_path):
    config = Config(
        work_dir=tmp_path,
        tokenizer=Tokens(kind="approx", estimated=True),
        parsing=Parsing(artifacts_path=os.environ["FINANCE_DOCLING_ARTIFACTS"]),
    )
    with Store(config.work_dir) as store:
        create_fixture(config, store, structured_snapshot=False)
        result = parse(config, store, "structured")
        assert result["completed"] == ["fictional-bank"] and not result["failed"]
        parsed = active_parses(config, store, "structured")[0]
        nodes = store.nodes(parsed["id"])
        table = next(n for n in nodes if n["kind"] == "table")
        assert len(table["headings"]) >= 2
        assert "2024" in table["text"]
        assert {s["page_index"] for n in nodes for s in n["sources"]} == {0, 1, 2}
        assert all(s["page_number"] == s["page_index"] + 1 for n in nodes for s in n["sources"])
        result = chunk(config, store, "structured")
        assert result["types"]["table"] > 0 and result["types"]["picture"] > 0
