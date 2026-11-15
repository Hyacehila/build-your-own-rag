from financial_rag.document.provenance import localize_fragment


def test_fragment_sources_do_not_inherit_unread_pages_or_guess_repeated_text():
    sources = [{"doc_version": "d", "page_index": i, "page_number": i + 1} for i in (0, 1)]
    node = {
        "id": "n",
        "text": "alpha beta",
        "sources": sources,
        "raw": {
            "orig": "alpha beta",
            "prov": [{"page_no": 1, "charspan": [0, 5]}, {"page_no": 2, "charspan": [6, 10]}],
        },
    }
    chunk = {"text": "Heading\nbeta", "headings": ["Heading"], "kind": "text", "sources": sources}
    result = localize_fragment(chunk, [node])
    assert result["sources"] == [sources[1]] and result["node_sources"] == sources
    assert result["fragment_spans"][0]["orig_start"] == 6
    repeated = {
        **node,
        "text": "beta beta",
        "raw": {
            "orig": "beta beta",
            "prov": [{"page_no": 1, "charspan": [0, 4]}, {"page_no": 2, "charspan": [5, 9]}],
        },
    }
    assert localize_fragment(chunk, [repeated])["source_alignment"] == "not_uniquely_aligned"
    assert chunk["sources"] == sources
