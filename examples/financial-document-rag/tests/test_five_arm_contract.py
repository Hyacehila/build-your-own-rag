import json

import pytest

from financial_rag.api import ModelAPI
from financial_rag.chunking import chunk, current_index
from financial_rag.common import RagError, digest
from financial_rag.config import ARMS, Query
from financial_rag.enrichment import enrich
from financial_rag.evaluation import evaluate
from financial_rag.retrieval import embed_index
from financial_rag.runner import QuestionRunner, run
from financial_rag.sources import Reader
from financial_rag.summary_windows import window_slices


@pytest.mark.parametrize('n,expected',[(0,[]),(1,[(0,1)]),(4,[(0,4)]),(5,[(0,5)]),
 (6,[(0,5),(4,6)]),(9,[(0,5),(4,9)]),(10,[(0,5),(4,9),(8,10)]),(13,[(0,5),(4,9),(8,13)])])
def test_windows(n,expected):
    assert list(window_slices(n))==expected
    assert set(i for a,b in expected for i in range(a,b))==set(range(n))


def prepare(config, store):
    assert not enrich(config,store)['failed']
    chunk(config,store,'enriched')
    for k in ["flat", "structured", "enriched"]:
        embed_index(config, store, k)


def test_five_arms_snapshots_indices_and_no_success_regeneration(configured):
    c, s, f = configured
    prepare(c, s)
    result=run(c,s,'five',list(ARMS))
    assert result['records']==15 and result['failures']==0
    rows = s.records("five")
    by = {(r["arm"], r["query_id"]): r for r in rows}
    for r in rows:
        if r["arm"] in ["A", "B", "C", "pre-D"]:
            assert r["model_calls"] == 1
        if r['arm'] in ['A','B','C']:
            assert sum(t['tool']=='search' for t in r['trace'])==1
        if r['arm'] in ['D','pre-D']:
            assert r['index_id']==by['B',r['query_id']]['index_id']
            assert r['initial_snapshot_sha256']==by['D',r['query_id']]['initial_snapshot_sha256']
    before = digest(rows)
    requests = len(f.requests)
    run(c,s,'five',list(ARMS),retry_errors=True)
    assert digest(s.records('five'))==before and len(f.requests)==requests
    assert evaluate(c,s,'five')['pending']==0
    # The maintained plan/report integration is exercised in test_experiment_setup.
    assert digest(s.records('five')) == before


def test_d24_explorations_then_one_forced_answer_all_tools_processed(configured):
    c, s, f = configured
    embed_index(c, s, "structured")

    def endless(path,body):
        if "tools" not in body:
            return None
        return 200,{'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
          'tool_calls':[{'id':f'call_{i}','type':'function','function':{'name':'search','arguments':'{"query":"fictional revenue"}'}} for i in range(6)]}}]}
    f.override=endless
    w=QuestionRunner(c,s,current_index(c,s,'structured'),ModelAPI(c,s,'twentyfour'))
    r=w.run({'query_id':'fiction','query':'fictional revenue'},'D')
    assert r['status']=='ok',r['error']
    assert r['model_calls']==25
    # Initial rewrite is exploration1; explorations2..24 each execute ALL six searches.
    assert sum(t['tool']=='search' for t in r['trace'])==1+23*6
    calls=[b for path,b in f.requests if path.endswith('/chat/completions')]
    assert len(calls)==25 and 'tools' not in calls[-1]


def test_d_rejects_enriched_and_capacity_must_be_attested(configured):
    c, s, f = configured
    prepare(c, s)
    w=QuestionRunner(c,s,current_index(c,s,'enriched'),ModelAPI(c,s,'wrong-index'))
    r=w.run({'query_id':'fiction','query':'fiction'},'D')
    assert r['status']=='failed' and 'structured original' in r['error']['message']
    c.query.capacity_evidence_path=None
    with pytest.raises(RagError, match="capacity"):
        run(c, s, "no-capacity", list(ARMS))


def test_old_switches_rejected():
    for key in ['autonomous_c','incremental_d','agent_tool_calls','autonomous_context_tokens']:
        with pytest.raises(ValueError):
            Query(**{key: True})


def test_user_declared_capacity_is_explicit_not_claimed_provider_verified(configured):
    from financial_rag.context_capacity import Capacity
    c,s,f=configured
    p = c.query.capacity_evidence_path
    proof = json.loads(p.read_text())
    proof.update(verification='user_declared',context_window_tokens=1000000,
                 image_counting='serialized_payload_tokens_proxy',provider_capacity_independently_verified=False)
    p.write_text(json.dumps(proof))
    cap=Capacity(c)
    messages=[{'role':'system','content':'source only'},{'role':'user','content':'question'},
              {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}} for _ in range(10)]}]
    packed=cap.fit(messages,None,[])
    assert sum(x['type']=='image_url' for x in packed[-1]['content'])==8
    assert not cap.identity['provider_capacity_independently_verified']
    assert cap.proof['context_window_tokens']==1000000


def test_same_originals_produce_identical_B_C_packets(configured):
    from financial_rag.evidence_packet import OriginalCatalog

    c, s, _ = configured
    prepare(c, s)
    base = current_index(c, s, "structured")
    enriched = current_index(c, s, "enriched")
    catalog=OriginalCatalog(s,base)
    source=next(x for x in s.chunks(base['id']) if x['kind']=='text')
    copied=next(x for x in s.chunks(enriched['id']) if not x.get('enrichment_id') and x['source_chunk_ids']==[source['id']])
    b = Reader(c, s, base)
    other = Reader(c, s, base)
    catalog.assemble(b, [source], base)
    catalog.assemble(other, [copied], enriched)
    assert b.content()==other.content()
    summary=next(x for x in s.chunks(enriched['id']) if x['kind']=='body_summary')
    r=Reader(c,s,base)
    result=catalog.assemble(r,[summary,summary],enriched)
    assert result['anchor_ids']==summary['source_chunk_ids']
    assert len(result['anchor_ids'])==len(set(result['anchor_ids']))
    assert all(e.get('chunk_id')!=summary['id'] for e in r.evidence.values())


def test_section_windows_do_not_cross_real_section_ids():
    from financial_rag.summary_windows import ordered_sections

    nodes = {}
    chunks = []
    for i in range(11):
        section='first' if i<6 else 'second'
        nid=f'n{i}'
        nodes[nid]={'id':nid,'kind':'text','parse_id':'p','section_id':section,'reading_order':i,'text':str(i)}
        chunks.append({'id':f'c{i}','kind':'text','node_ids':[nid],'text':str(i),'headings':['Repeated title']})
    groups=ordered_sections(chunks,nodes)
    assert [len(v) for v in groups.values()]==[6,5]
    assert [list(window_slices(len(v))) for v in groups.values()]==[[(0,5),(4,6)],[(0,5)]]


@pytest.mark.parametrize('succeed_on',[2,3,4])
def test_success_on_each_recovery_round_stops_future_generation(configured,succeed_on):
    c, s, _ = configured
    embed_index(c, s, "flat")
    from unittest.mock import patch
    original=QuestionRunner.run
    counter=0
    def controlled(self,q,arm,**kw):
        nonlocal counter
        counter+=1
        if counter<succeed_on:
            return {**q,'arm':arm,'status':'failed','answer':None,'citations':[],'error':{'type':'SyntheticError'},
                    'evidence':[],'initial_page_ranking':[],'trace':[],'api_usage':{},'index_id':self.index['id']}
        return original(self,q,arm,**kw)
    with patch.object(QuestionRunner,'run',controlled):
        run(c,s,'round-success',['A'],limit=1)
        for k in range(1, 4):
            run(c, s, "round-success", ["A"], limit=1, retry_errors=True, recovery_round=k)
        assert counter==succeed_on
        assert s.records('round-success')[0]['status']=='ok'


def test_unbounded_original_read_has_no_hidden_40k_or_96k(configured):
    c,s,_=configured
    reader=Reader(c,s,current_index(c,s,'structured'),unbounded=True,image_limit=16)
    reader._read_limit=float('inf')
    source=next(iter(s.chunks(reader.index['id'])))['sources']
    text=' original factual sentence'*12000
    r=reader._text('synthetic-large-source',text,source,0,'text')
    assert reader.evidence[r['evidence_id']]['text']==text
    assert reader.text_used>40000
    assert reader.remaining==float('inf')


def test_uniform_three_retries_preserve_success_and_count_exhaustion_false(configured):
    c, s, f = configured
    embed_index(c, s, "flat")
    # Successful question is frozen before introducing a transport failure for the other.
    original=QuestionRunner.run
    def controlled(self,q,arm,**kw):
        if q['query_id']=='q2':
            return {**q,'arm':arm,'status':'failed','answer':None,'citations':[],
                    'error':{'type':'SyntheticError','message':'fixture'},'evidence':[],
                    'initial_page_ranking':[],'trace':[],'api_usage':{},'index_id':self.index['id']}
        return original(self,q,arm,**kw)
    from unittest.mock import patch
    with patch.object(QuestionRunner,'run',controlled):
        run(c,s,'retry-four',['A'])
        success={r['query_id']:digest(r) for r in s.records('retry-four') if r['status']=='ok'}
        for _ in range(3):
            run(c, s, "retry-four", ["A"], retry_errors=True)
        records=s.records('retry-four')
        for r in records:
            if r["status"] == "ok":
                assert digest(r) == success[r["query_id"]]
            else:
                assert r["attempt"] == 4 and len(r["previous_attempts"]) == 3
        before = digest(records)
        run(c, s, "retry-four", ["A"], retry_errors=True)
        assert digest(s.records('retry-four'))==before


def test_unbounded_tool_response_is_strict_json_and_reports_actual_image_limit(configured):
    from financial_rag.common import canonical

    c, s, _ = configured
    base = current_index(c, s, "structured")
    reader=Reader(c,s,base,unbounded=True,image_limit=16)
    reader.images={f'fictional-image-{i}':'retained' for i in range(9)}
    block=next(x for x in s.chunks(base['id']) if x['kind']=='text')
    result=reader.read(chunk_id=block['id'],include_images=False,mode='node')
    canonical(result)
    canonical(reader.trace)
    assert result['remaining_text_tokens'] is None
    assert result['remaining_page_images']==7
    assert reader.remaining==float('inf')


def test_d_original_read_tool_roundtrip_and_record_persistence(configured):
    from financial_rag.common import canonical

    c, s, f = configured
    embed_index(c, s, "structured")
    base = current_index(c, s, "structured")
    block=next(x for x in s.chunks(base['id']) if x['kind']=='text')
    issued=False
    def read_once(path,body):
        nonlocal issued
        if "tools" not in body or issued:
            return None
        issued=True
        return 200,{'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
            'tool_calls':[{'id':'fixture-read-original','type':'function','function':{'name':'read',
              'arguments':json.dumps({'source_ids':[block['id']],'include_images':False})}}]}}]}
    f.override=read_once
    run(c,s,'read-roundtrip',['D','pre-D'],limit=1)
    records = s.records("read-roundtrip")
    assert len(records) == 2
    assert all(r['status']=='ok' for r in records),[(r['arm'],r['error']) for r in records]
    d=next(r for r in records if r['arm']=='D')
    assert d['model_calls']==3
    trace=next(t for t in d['trace'] if t.get('actor')=='model' and t['tool']=='read')
    assert trace['response']['remaining_text_tokens'] is None
    canonical(records)
    response_messages=[m for path,b in f.requests if path.endswith('/chat/completions')
                       for m in b['messages'] if m['role']=='tool' and m.get('tool_call_id')=='fixture-read-original']
    assert response_messages and json.loads(response_messages[-1]['content'])['remaining_text_tokens'] is None




def test_D_initial_packet_has_no_controller_context_expansion(configured):
    c, s, f = configured
    embed_index(c, s, "structured")
    w=QuestionRunner(c,s,current_index(c,s,'structured'),ModelAPI(c,s,'initial-no-auto-expansion'))
    row=w.run({'query_id':'fiction','query':'Fictional revenue'},'D')
    assert row['status']=='ok',row['error']
    assert w.initial_snapshot['packet']['assembly']['neighbor_ids']==[]
    assert not any(e.get('kind')=='page' for e in w.initial_snapshot['packet']['evidence'].values())


def test_driver_rejects_unapproved_full_before_loading_config(tmp_path):
    import subprocess
    import sys

    p = tmp_path / "plan.json"
    p.write_text(
        json.dumps({"stages": [{"query_ids": [str(i) for i in range(296)]}], "full_run_authorized": False})
    )
    result=subprocess.run([sys.executable,'tools/contract_experiment.py','--plan',str(p)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Full experiment requires a new explicit user decision' in result.stderr
