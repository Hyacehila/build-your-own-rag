from types import SimpleNamespace

import pytest

from financial_rag.indexing.chunking import current_index
from financial_rag.query.bc_restoration import (
 assemble_restored,
 base_coverage,
 reference_targets,
 select_coverage,
 select_spans,
)
from financial_rag.query.evidence_packet import OriginalCatalog
from financial_rag.query.structural_expansion import StructuralReader, Structure


def span(node='n',start=0,end=10,version='v'):
 return {'doc_version':version,'parse_id':'p','source_id':node,'representation':'original_node_text','start':start,'end':end}

def hit(cid,score=1):return {'id':cid,'score':score,'kind':'text'}


def test_exact_coverage_skip_partial_fragments_and_refill():
 candidates=[(hit('a'),[span()]),(hit('b'),[span()]),(hit('c'),[span(start=5,end=15)]),(hit('d'),[span(node='other')])]
 out,log=select_spans(candidates,3)
 assert [x['id'] for x in out]==['a','c','d']
 assert log[1]['reason']=='fully_covered'
 assert [(s['start'],s['end']) for s in log[2]['new_coverage']]==[(10,15)]
 assert out[0]['score']==1  # Duplicate entrances never boost score.


def test_same_node_different_version_not_deduplicated_and_short_pool():
 out,_=select_spans([(hit('a'),[span()]),(hit('b'),[span(version='v2')])],20)
 assert len(out)==2
 out,log=select_spans([(hit('a'),[]),(hit('b'),[])],20)
 assert len(out)==2 and all(x['reason']=='unknown_coverage_retained' for x in log)


def test_b_c_same_base_sources_identical_and_regular_pages_without_gaps(configured):
    c, store, _ = configured
    base = current_index(c, store, "structured")
    cat = OriginalCatalog(store, base)
    hits = [next(x for x in cat.chunks.values() if x["kind"] == "table")] + [
        x for x in cat.chunks.values() if x["kind"] == "text"
    ][:2]
    r1 = StructuralReader(c, store, base)
    r1.question = "fictional revenue"
    r2 = StructuralReader(c, store, base)
    r2.question = "fictional revenue"
    a = assemble_restored(r1, hits, base, cat)
    assemble_restored(r2, hits, base, cat)
    assert r1.public_evidence() == r2.public_evidence()
    assert a["contract"] == "BC-reference-context-v2"
    assert [phase["phase"] for phase in a["phases"]] == ["originals", "references", "local_context"]
    assert r1.images  # The table is tied to the actual source page.
    assert r1.text_used <= 25000 and len(r1.images) <= 8
    assert c.query.text_tokens == 40000 and r1.config.query.text_tokens == 25000


def test_original_coverage_equivalent_summary_refill(configured):
    c, store, _ = configured
    base = current_index(c, store, "structured")
    cat = OriginalCatalog(store, base)
    hits = list(cat.chunks.values())[:4]
    one = hits[0]
    # Original identity resolver is unchanged; synthetic wrapper simulates summary binding.
    summary = {**one, "id": "summary", "source_chunk_ids": [one["id"]], "enrichment_id": "fiction"}
    out, log = select_coverage([one, summary, *hits[1:]], 20, cat, base)
    assert log[1]["reason"] == "fully_covered"
    assert base_coverage(one, cat, base) == base_coverage(summary, cat, base)


def test_budget_stops_are_explicit_and_no_page_recursion(configured):
    c, store, _ = configured
    c.query.text_tokens = 30
    base = current_index(c, store, "structured")
    cat = OriginalCatalog(store, base)
    r = StructuralReader(c, store, base)
    r.question = "fictional"
    out = assemble_restored(r, list(cat.chunks.values())[:10], base, cat)
    assert r.text_used <= 30 and out["unread_ranges"]
    assert any(g["unread_node_ranges"] for g in out["resolved_groups"])


def test_runner_restoration_only_B_C_and_RRF20(configured):
    from financial_rag.query.runner import run

    c, store, f = configured
    c.retrieval.rrf_k = 20
    from test_five_arm_contract import prepare

    prepare(c, store)
    run(c, store, "restoration-fixture", ["A", "B", "C", "D", "pre-D"], limit=1)
    rows = store.records("restoration-fixture")
    assert all(r["status"] == "ok" for r in rows), [(r["arm"], r["error"]) for r in rows]
    for r in rows:
        searches = [t for t in r["trace"] if t.get("tool") == "search"]
        if r["arm"] in ["B", "C"]:
            assert len(searches) == 1 and r["model_calls"] == 1
            assert searches[0]["rrf_k"] == 20 and "coverage_selection" in searches[0]
            for entry in searches[0]["coverage_selection"]:
                expected = sum(
                    1 / (20 + entry[k]) for k in ["sparse_rank", "dense_rank"] if entry[k] is not None
                )
                assert entry["rrf_score"] == pytest.approx(expected)
            assert r["fixed_assembly"]["contract"] == "BC-reference-context-v2"
        else:
            assert all("coverage_selection" not in t for t in searches)
    d = next(r for r in rows if r["arm"] == "D")
    p = next(r for r in rows if r["arm"] == "pre-D")
    assert d["initial_snapshot_sha256"] == p["initial_snapshot_sha256"]
    assert d["initial_expansion"]["window_policy"] == "D-fixed-eight-v1"
    from financial_rag.evaluation import evaluate

    assert evaluate(c, store, "restoration-fixture")["pending"] == 0
    assert all(
        r["fixed_assembly"]["contract"] == "BC-reference-context-v2" for r in rows if r["arm"] in ("B", "C")
    )


def test_reference_targets_require_unique_scoped_original_and_map_caption_to_table():
    def node(nid, kind, text, section, parent, order, parse="p", related=()):
        return {"id": nid, "ref": nid, "kind": kind, "text": text, "section_id": section,
                "parent_id": parent, "reading_order": order, "parse_id": parse, "related_refs": list(related),
                "sources": [{"doc_version": parse, "page_index": 0}]}

    items = [
        node("root", "section_header", "Report", "root", None, 0),
        node("s1", "section_header", "One", "s1", "root", 1),
        node("anchor", "text", "See Table 9 and Figure 3 and Table 10.", "s1", "s1", 2),
        node("s2", "section_header", "Two", "s2", "root", 3),
        node("table", "table", "Revenue 2024", "s2", "s2", 4, related=["caption"]),
        node("caption", "caption", "Table 9: Revenue", "s2", "s2", 5),
        node("figure1", "picture", "Figure 3", "s2", "s2", 6),
        node("figure2", "picture", "Figure 3", "s2", "s2", 7),
        node("other_root", "section_header", "Other", "other_root", None, 8),
        node("other_section", "section_header", "Other section", "other_section", "other_root", 9),
        node("outside", "table", "Table 10", "other_section", "other_section", 10),
        node("foreign_root", "section_header", "Foreign", "foreign_root", None, 0, parse="other"),
        node("foreign_section", "section_header", "Foreign section", "foreign_section", "foreign_root", 1, parse="other"),
        node("foreign", "table", "Table 9", "foreign_section", "foreign_section", 2, parse="other"),
    ]
    nodes = {n["id"]: n for n in items}
    catalog = SimpleNamespace(nodes=nodes, by_ref={(n["parse_id"], n["ref"]): n for n in items})
    structure = Structure(catalog, SimpleNamespace(count=lambda text: len(text.split())))
    targets, resolved, gaps = reference_targets(structure, ["anchor"])
    assert targets == ["table"]
    assert resolved == [{"anchor": "anchor", "reference": ["table", "9"], "target": "table"}]
    assert [(gap["reference"], gap["reason"]) for gap in gaps] == [
        (["figure", "3"], "reference_ambiguous"), (["table", "10"], "reference_unresolved")
    ]


def test_b_and_c_read_unique_reference_before_local_context(configured):
    from copy import deepcopy

    from financial_rag.common import canonical

    c, store, _ = configured
    base = current_index(c, store, "structured")
    catalog = OriginalCatalog(store, base)
    hit = next(x for x in catalog.chunks.values() if x["kind"] == "text")
    anchor = deepcopy(catalog.nodes[hit["node_ids"][0]])
    table = deepcopy(next(n for n in catalog.nodes.values() if n["kind"] == "table" and n["parse_id"] == anchor["parse_id"]))
    for nid, parent, order in [("reference-root", None, 10000), ("reference-s1", "reference-root", 10001),
                               ("reference-s2", "reference-root", 10003)]:
        heading = {**anchor, "id": nid, "ref": nid, "kind": "section_header", "text": nid,
                   "section_id": nid, "parent_id": parent, "reading_order": order, "related_refs": []}
        store.db.execute("INSERT INTO nodes(id, parse_id, ref, payload) VALUES (?, ?, ?, ?)",
                         (nid, heading["parse_id"], nid, canonical(heading)))
    anchor.update(text="Please see Table 9.", section_id="reference-s1", parent_id="reference-s1", reading_order=10002)
    table.update(text="Table 9\nRevenue: 42", section_id="reference-s2", parent_id="reference-s2", reading_order=10004,
                 related_refs=[])
    for node in (anchor, table):
        store.db.execute("UPDATE nodes SET payload=? WHERE id=?", (canonical(node), node["id"]))
    catalog = OriginalCatalog(store, base)
    reader = StructuralReader(c, store, base)
    result = assemble_restored(reader, [hit], base, catalog)
    assert result["references"] == [{"anchor": anchor["id"], "reference": ["table", "9"], "target": table["id"]}]
    assert table["id"] in result["events"][1]["selected_node_ids"]
    assert any("Revenue: 42" in evidence.get("text", "") for evidence in reader.public_evidence())
    assert result["events"][1]["new_evidence"]


def test_b_and_c_use_configured_retrieval_limits(configured):
    from test_five_arm_contract import prepare

    from financial_rag.query.runner import run

    config, store, _ = configured
    prepare(config, store)
    config.retrieval.rrf_k = 7
    config.retrieval.sparse_k = 3
    config.retrieval.dense_k = 4
    config.retrieval.final_k = 2
    assert run(config, store, "custom-retrieval", ["B", "C"], limit=1)["failures"] == 0
    for row in store.records("custom-retrieval"):
        search = next(event for event in row["trace"] if event.get("tool") == "search")
        original = search.get("original_search", search)
        assert original["rrf_k"] == 7
        assert original["final_k"] == 2
        assert len(original["sparse"]) <= 3 and len(original["dense"]) <= 4


def test_rrf_stable_tie_break_with_and_without_coverage_selection(configured):
    from financial_rag.api import ModelAPI
    from financial_rag.query.retrieval import Search, embed_index

    c, store, _ = configured
    c.retrieval.rrf_k = 20
    embed_index(c, store, "structured")
    idx = current_index(c, store, "structured")
    cat = OriginalCatalog(store, idx)
    search = Search(c, store, idx, ModelAPI(c, store, "fixture-rrf"))
    search.search("fictional assets", coverage_selector=lambda hits, k: select_coverage(hits, k, cat, idx))
    log = search.trace[-1]
    pairs = [(x["rrf_score"], x["candidate_id"]) for x in log["coverage_selection"]]
    assert pairs == sorted(pairs, key=lambda x: (-x[0], x[1]))
    search.search("fictional assets")
    assert "coverage_selection" not in search.trace[-1]


def test_enriched_entrance_without_summary_same_B_packet_and_summary_full_members(configured):
    from test_five_arm_contract import prepare

    c, store, _ = configured
    prepare(c, store)
    base = current_index(c, store, "structured")
    enriched = current_index(c, store, "enriched")
    cat = OriginalCatalog(store, base)
    original = next(x for x in cat.chunks.values() if x["kind"] == "text")
    equivalent = next(
        x
        for x in store.chunks(enriched["id"])
        if not x.get("enrichment_id") and x.get("source_chunk_ids") == [original["id"]]
    )
    left = StructuralReader(c, store, base)
    right = StructuralReader(c, store, base)
    left.question = right.question = "fictional revenues"
    assemble_restored(left, [original], base, cat)
    assemble_restored(right, [equivalent], enriched, cat)
    assert left.public_evidence() == right.public_evidence()
    summary = next(x for x in store.chunks(enriched["id"]) if x["kind"] == "body_summary")
    expected = set(n for cid in summary["source_chunk_ids"] for n in cat.chunks[cid]["node_ids"])
    r = StructuralReader(c, store, base)
    r.question = "fictional revenue"
    result = assemble_restored(r, [summary], enriched, cat)
    assert expected <= set(result["anchor_node_ids"])
    assert all(e.get("source_key") != "chunk:" + summary["id"] for e in r.public_evidence())
