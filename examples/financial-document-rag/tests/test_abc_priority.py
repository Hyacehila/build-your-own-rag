"""Fictional regressions for the A, B and C evidence-selection strategies."""
from financial_rag.api import ModelAPI
from financial_rag.chunking import current_index
from financial_rag.evidence_packet import OriginalCatalog
from financial_rag.retrieval import Search
from financial_rag.runner import QuestionRunner
from financial_rag.bc_restoration import select_supplement
from test_five_arm_contract import prepare


def test_a_selection_is_ranked_prefix_without_changing_candidates(configured):
    cfg,store,_=configured;prepare(cfg,store)
    api=ModelAPI(cfg,store,'fictional-top10');index=current_index(cfg,store,'flat')
    search=Search(cfg,store,index,api)
    twenty=search.search('fictional revenue',final_k=20);before=search.trace[-1]
    ten=search.search('fictional revenue',final_k=10);after=search.trace[-1]
    assert [x['id'] for x in ten]==[x['id'] for x in twenty[:10]]
    assert before['sparse']==after['sparse'] and before['dense']==after['dense']
    cfg.retrieval.flat_final_k=10
    worker=QuestionRunner(cfg,store,index,api,search)
    worker.direct('fictional revenue')
    assert worker.trace[0]['final_k']==10


def test_c_zero_supplements_exactly_reproduces_b_original_packet(configured):
    cfg,store,_=configured;prepare(cfg,store);cfg.retrieval.summary_supplement_k=0
    base=current_index(cfg,store,'structured');enriched=current_index(cfg,store,'enriched')
    b=QuestionRunner(cfg,store,base,ModelAPI(cfg,store,'B-zero'))
    c=QuestionRunner(cfg,store,enriched,ModelAPI(cfg,store,'C-zero'),original_index=base,catalog=OriginalCatalog(store,base))
    b.direct('fictional revenue',expanded_BC=True);c.direct('fictional revenue',expanded_BC=True)
    assert b.reader.public_evidence()==c.reader.public_evidence()
    assert b.assembly['anchor_ids']==c.assembly['anchor_ids']
    assert c.trace[0]['summary_supplement_ids']==[]
    assert b.assembly['phases']==c.assembly['phases']


def test_summary_never_displaces_originals_and_consumes_no_second_embedding(configured):
    cfg,store,fake=configured;prepare(cfg,store)
    base=current_index(cfg,store,'structured');enriched=current_index(cfg,store,'enriched')
    c=QuestionRunner(cfg,store,enriched,ModelAPI(cfg,store,'C-protection'),original_index=base,catalog=OriginalCatalog(store,base))
    n=sum(path.endswith('/embeddings') for path,_ in fake.requests)
    c.direct('new fictional original protection query',expanded_BC=True)
    assert sum(path.endswith('/embeddings') for path,_ in fake.requests)-n==1
    t=c.trace[0];primary=t['protected_original_ids']
    assert primary==[x['id'] for x in t['original_search']['ranked_chunks']]
    assert [x['id'] for x in t['ranked_chunks'][:len(primary)]]==primary
    assert len(t['summary_supplement_ids'])<=5
    assert c.model_calls==1 and c.tool_calls==1
    b=QuestionRunner(cfg,store,base,ModelAPI(cfg,store,'B-protection'))
    b.direct('new fictional original protection query',expanded_BC=True)
    count=b.assembly['phases'][0]['evidence_count']
    assert c.reader.public_evidence()[:count]==b.reader.public_evidence()[:count]
    assert [p['phase'] for p in c.assembly['phases']]==[
        'anchors_and_related','explicit_relations','local_structure','parent_overview',
        'associated_sections_and_cross_heading','bound_original_pages']


def test_duplicate_summary_cannot_consume_slot_even_when_highest_score(configured):
    cfg,store,_=configured;prepare(cfg,store)
    base=current_index(cfg,store,'structured');cat=OriginalCatalog(store,base)
    primary=next(x for x in cat.chunks.values() if x['kind']=='text')
    duplicate={**primary,'id':'fictional-summary','enrichment_id':'summary','source_chunk_ids':[primary['id']],'score':99}
    selected,log=select_supplement([duplicate],5,cat,base,[primary])
    assert selected==[] and log[0]['reason']=='fully_covered'
