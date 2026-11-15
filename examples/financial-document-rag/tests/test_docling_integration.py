"""Opt-in parsing check against the real upstream financial document."""

import os
import shutil
from pathlib import Path

import pymupdf
import pytest

from financial_rag.config import Config, Parsing, Tokens
from financial_rag.document.parsing import _converter, docling_nodes


@pytest.mark.docling
@pytest.mark.skipif(
    not os.environ.get("FINANCE_DOCLING_ARTIFACTS") or not os.environ.get("FINANCE_TEST_PDF"),
    reason="Set FINANCE_TEST_PDF to the upstream JPMorgan report and provide local Docling weights",
)
def test_real_pdf_parser_preserves_table_and_physical_page(tmp_path):
    pdf_path = tmp_path / "jpmorgan_chase_2024.pdf"
    shutil.copy2(Path(os.environ["FINANCE_TEST_PDF"]), pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        assert len(pdf) >= 220
        assert "15,371" in pdf[219].get_text() and "14,655" in pdf[219].get_text()
        pages = len(pdf)
    config = Config(
        work_dir=tmp_path,
        tokenizer=Tokens(kind="approx", estimated=True),
        parsing=Parsing(artifacts_path=os.environ["FINANCE_DOCLING_ARTIFACTS"]),
    )
    doc = {"id": "real-report-test", "doc_id": "jpmorgan_chase_2024",
           "filename": pdf_path.name, "pdf_path": str(pdf_path), "pages": pages}
    result = _converter(config).convert(pdf_path, page_range=(220, 220), raises_on_error=False)
    assert result.status.value == "success"
    nodes = docling_nodes(result.document, doc, "integration-test")
    tables = [node for node in nodes if node["kind"] == "table"]
    assert any("15,371" in node["text"] and "14,655" in node["text"] for node in tables)
    assert all(source["page_number"] == 220 for node in nodes for source in node["sources"])
