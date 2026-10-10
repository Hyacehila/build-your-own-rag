import json
from types import SimpleNamespace
import pytest
from financial_rag.structural_expansion import Structure,StructuralReader,read_nodes
from financial_rag.runner import validate_coverage, QuestionRunner
from financial_rag.tokenization import Tokenizer
from financial_rag.config import Tokens
from financial_rag.chunking import current_index
from financial_rag.evidence_packet import OriginalCatalog


def graph(length=30,long=False):
    nodes={}
    for i in range(length):
        nodes[str(i)]={'id':str(i),'parse_id':'p','section_id':'s','reading_order':i,'kind':'text',
                      'text':('fictional prose '*1000 if long else f'fictional paragraph {i}'),'related_refs':[]}
    cat=SimpleNamespace(nodes=nodes,by_ref={},chunks={})
    return Structure(cat,Tokenizer(Tokens(kind='approx',name='test',estimated=True)))


def test_eight_body_nodes_each_side_assets_do_not_consume_body_positions():
    s=graph(32,long=True);s.nodes['15']['kind']='table'
    chosen=s.select(['16'],radius=8)
    assert chosen==[str(i) for i in range(7,25)]
    assert '15' in chosen and len([n for n in chosen if s.nodes[n]['kind']=='text'])==17
    # No expansion from newly included outermost nodes.
    assert '6' not in chosen and '25' not in chosen


def test_section_boundary_stops_without_returning_after_child():
    s=graph();s.nodes['10'].update(kind='section_header',section_id='child')
    for i in range(11,15):s.nodes[str(i)]['section_id']='child'
    s=Structure(s.catalog,s.tokenizer)
    assert s.select(['9'],radius=8)==[str(i) for i in range(1,10)]
    assert s.select(['9'],whole=True)==[str(i) for i in range(10)]
    assert '15' not in s.select(['9'],whole=True)


def test_short_section_whole_and_long_section_window_overlap_dedup():
    s=graph();assert len(s.select(['15'],short=True))==30
    s=graph(long=True);chosen=s.select(['14','15'],short=True)
    assert chosen==[str(i) for i in range(6,24)]
    assert len(chosen)==len(set(chosen))


def test_explicit_captions_and_unresolved_links_only_trigger_gaps():
    s=graph();s.nodes['0'].update(kind='table',related_refs=['cap']);s.nodes['1']['kind']='caption'
    s.by_ref[('p','cap')]=s.nodes['1']
    assert s.related(['0'])[0]==['0','1']
    assert not s.related(['0'])[1]
    s.nodes['0']['related_refs'].append('missing')
    assert s.related(['0'])[1][0]['reason']=='unresolved_explicit_relation'
    s.nodes['2'].update(kind='picture',text='')
    assert not s.related(['2'])[1]


def test_long_node_continues_beyond_2400_and_same_node_not_duplicated(configured):
    c,store,_=configured;base=current_index(c,store,'structured');cat=OriginalCatalog(store,base)
    n=next(n for n in cat.nodes.values() if n['kind']=='text' and n.get('sources'))
    n={**n,'text':'Fictional continuous original prose. '*1000};cat.nodes[n['id']]=n
    # Fictional test facts only, never formal source data.
    with store.db:store.db.execute('UPDATE nodes SET payload=? WHERE id=?',(json.dumps(n),n['id']))
    s=Structure(cat,Tokenizer(c.tokenizer));reader=StructuralReader(c,store,base)
    out=read_nodes(reader,s,[n['id'],n['id']],include_images=False)
    assert reader.text_used>2400 and not out['unread_ranges']
    assert sum(e['text'].count('Fictional') for e in reader.evidence.values())==1000


def test_table_truncates_on_rows_and_continues_without_duplicate_header(configured):
    c,store,_=configured;reader=StructuralReader(c,store,current_index(c,store,'structured'))
    sources=next(iter(store.chunks(reader.index['id'])))['sources']
    text='Column A | Column B\n'+'fictional row | 123\n'*100
    reader._read_limit=40
    out=reader._text('synthetic-table',text,sources,0,'table');e=reader.evidence[out['evidence_id']]
    assert e['text'].endswith('\n') and e['end_offset']<len(text)
    reader._read_limit=10000
    nxt=reader._text('synthetic-table',text,sources,e['end_offset'],'table')
    assert e['text']+reader.evidence[nxt['evidence_id']]['text']==text


def test_coverage_rejects_missing_duplicate_unread_and_accepts_honest_gaps():
    entry={'item_id':'R1','status':'supported','evidence_ids':['E1'],'scope_check':'Period and units checked','gaps':''}
    validate_coverage({'coverage':[entry]},[{'item_id':'R1'}],{'E1':{}})
    for entries in [[],[entry,entry],[{**entry,'evidence_ids':['E9']}],[{**entry,'status':'partial'}]]:
        with pytest.raises(ValueError):validate_coverage({'coverage':entries},[{'item_id':'R1'}],{'E1':{}})
    validate_coverage({'coverage':[{**entry,'status':'not_found','evidence_ids':[],'gaps':'Not located'}]},[{'item_id':'R1'}],{})


@pytest.mark.parametrize('name',['read','read_source_pages'])
def test_explicit_tools_roundtrip(configured,name):
    from financial_rag.retrieval import embed_index
    from financial_rag.runner import run
    c,store,f=configured;embed_index(c,store,'structured');base=current_index(c,store,'structured')
    chunk=next(x for x in store.chunks(base['id']) if x['kind']=='text')
    args={'source_ids':[chunk['id']],'include_images':False}
    issued=False
    def override(path,body):
        nonlocal issued
        if 'tools' not in body or issued:return None
        issued=True
        return 200,{'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
            'tool_calls':[{'id':'explicit-read','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}
    f.override=override
    run(c,store,'explicit-'+name,['D','pre-D'],limit=1)
    rows=store.records('explicit-'+name);assert all(r['status']=='ok' for r in rows),[(r['arm'],r['error']) for r in rows]
    d=next(r for r in rows if r['arm']=='D');assert not d['coverage_diagnostic']['required']
    t=next(t for t in d['trace'] if t.get('tool')==name);assert t['actor']=='model' and t['tool_call_id']=='explicit-read'
    assert 'error' not in t['response']


def test_original_index_boundary_for_node_tools(configured):
    c,store,_=configured;b=current_index(c,store,'structured');cat=OriginalCatalog(store,b)
    s=Structure(cat,Tokenizer(c.tokenizer));reader=StructuralReader(c,store,b,unbounded=True)
    from financial_rag.agent_reading import FixedReading
    with pytest.raises(Exception,match='Unknown original source_id'):FixedReading(reader,s,10000).read(['foreign-summary'])




def test_forced_final_without_coverage_succeeds_at_25(configured):
    from financial_rag.retrieval import embed_index
    from financial_rag.api import ModelAPI
    c,store,f=configured;embed_index(c,store,'structured');c.models.chat.json_retries=0
    def override(path,body):
        if 'messages' not in body:return None
        if 'tools' in body:
            return 200,{'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':'keep-looking','type':'function','function':{'name':'search','arguments':'{"query":"fictional revenue"}'}}]}}]}
        content=body['messages'][-1].get('content','')
        if isinstance(content,str) and content.startswith('Exploration has ended'):
            return 200,{'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'{"answer":"fiction","citations":["E1"]}'}}]}
    f.override=override
    worker=QuestionRunner(c,store,current_index(c,store,'structured'),ModelAPI(c,store,'forced-coverage-fail'))
    result=worker.run({'query_id':'fiction','query':'fictional revenue'},'D')
    assert result['model_calls']==25 and result['status']=='ok'
    assert not result['coverage_diagnostic']['required']
