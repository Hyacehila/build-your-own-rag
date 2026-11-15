from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

from financial_rag.common import write_json
from financial_rag.config import Chunking, Tokens


def test_hybrid_uses_fact_parents_at_independent_appendix_boundary(tmp_path):
    def provenance(page, text):
        return ProvenanceItem(
            page_no=page, charspan=(0, len(text)),
            bbox=BoundingBox(l=10, t=20, r=150, b=40, coord_origin=CoordOrigin.TOPLEFT),
        )

    from financial_rag.config import Config
    from financial_rag.document.parsing import docling_nodes
    from financial_rag.indexing.chunking import hybrid_chunks
    from financial_rag.tokenization import Tokenizer

    doc = DoclingDocument(name="two independent scopes")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    first = doc.add_heading(text="Previous exhibit", level=1, prov=provenance(1, "Previous exhibit"))
    doc.add_text(label=DocItemLabel.TEXT, text="Old body", parent=first, prov=provenance(1, "Old body"))
    # Level 2 does not imply a child: an independently reviewed scope owns this heading.
    group = doc.add_group(name="source-range:appendix")
    second = doc.add_heading(text="New appendix", level=2, parent=group, prov=provenance(1, "New appendix"))
    doc.add_text(
        label=DocItemLabel.TEXT, text="New body " * 900, parent=second, prov=provenance(1, "New body " * 900)
    )
    path = tmp_path / "doc.json"
    write_json(path, doc.model_dump(mode="json"))
    nodes = docling_nodes(doc, {"doc_id": "scope", "id": "version", "pages": 1}, "parse")
    config = Config(
        tokenizer=Tokens(kind="approx", name="offline-test", estimated=True),
        chunking=Chunking(max_tokens=100, overlap=10),
    )
    tokenizer = Tokenizer(config.tokenizer)
    chunks = hybrid_chunks({"snapshot": str(path)}, nodes, config, tokenizer)
    current = [c for c in chunks if "New body" in c["text"]]
    assert len(current) > 1
    assert all(c["headings"] == ["New appendix"] and "Previous exhibit" not in c["text"] for c in current)
    assert all(tokenizer.count(c["text"]) <= 100 for c in chunks)
