import pytest

from financial_rag.indexing.chunking import current_index
from financial_rag.query.bc_restoration import (
 assemble_restored,
 base_coverage,
 select_coverage,
 select_spans,
)
from financial_rag.query.evidence_packet import OriginalCatalog
from financial_rag.query.structural_expansion import StructuralReader


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
    assert a["contract"] == "BC-local-context-v1"
    assert [phase["phase"] for phase in a["phases"]] == ["originals", "local_context"]
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
            assert r["fixed_assembly"]["contract"] == "BC-local-context-v1"
        else:
            assert all("coverage_selection" not in t for t in searches)
    d = next(r for r in rows if r["arm"] == "D")
    p = next(r for r in rows if r["arm"] == "pre-D")
    assert d["initial_snapshot_sha256"] == p["initial_snapshot_sha256"]
    assert d["initial_expansion"]["window_policy"] == "D-fixed-eight-v1"
    from financial_rag.evaluation import evaluate

    assert evaluate(c, store, "restoration-fixture")["pending"] == 0
    assert all(
        r["fixed_assembly"]["contract"] == "BC-local-context-v1" for r in rows if r["arm"] in ("B", "C")
    )


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
