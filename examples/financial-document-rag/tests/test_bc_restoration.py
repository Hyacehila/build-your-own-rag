from types import SimpleNamespace
import pytest
from financial_rag.bc_restoration import (select_spans,select_coverage,base_coverage,Navigation,
                                         assemble_restored,read_bound_page)
from financial_rag.structural_expansion import Structure,StructuralReader
from financial_rag.tokenization import Tokenizer
from financial_rag.config import Tokens
from financial_rag.chunking import current_index
from financial_rag.evidence_packet import OriginalCatalog


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


def nav_graph():
 nodes={};order=0
 def add(nid,kind,parent,text='',section=None):
  nonlocal order
  nodes[nid]={'id':nid,'parse_id':'p','kind':kind,'parent_id':parent,'section_id':section,
              'reading_order':order,'text':text or nid,'related_refs':[],'sources':[]};order+=1
 add('P','section_header',None,'Risk')
 add('intro','text','P','parent introduction',section='P')
 add('S1','section_header','P','Consumer credit')
 for i in range(12):add(f'a{i}','text','S1',('a long fictional paragraph '*250),section='S1')
 add('S2','section_header','P','Corporate credit')
 for i in range(12):add(f'b{i}','text','S2',('another long fictional paragraph '*250),section='S2')
 add('sub','section_header','S2','Details')
 add('subbody','text','sub','nested text',section='sub')
 add('Q','section_header',None,'Other chapter')
 add('S3','section_header','Q','Credit elsewhere')
 add('outside','text','S3','no',section='S3')
 cat=SimpleNamespace(nodes=nodes,by_ref={},chunks={})
 return Navigation(Structure(cat,Tokenizer(Tokens(kind='approx',name='test',estimated=True))))


def test_parent_overview_only_before_first_child():
 n=nav_graph();assert n.direct('P')==['intro']
 assert n.parent_of_anchor('a11')=='P'


def test_default_cross_sibling_window_no_title_match_required():
 n=nav_graph();window,parent=n.adjacent('a11')
 assert parent=='P' and 'b0' in window and 'b7' in window and 'b8' not in window
 assert 'a2' not in window and 'a3' in window
 assert 'outside' not in window


def test_no_climb_beyond_nearest_parent_and_long_section_eight():
 n=nav_graph();window,parent=n.adjacent('subbody')
 assert parent=='S2' and 'a11' not in window
 assert n.bounded_section('S2')==[f'b{i}' for i in range(8)]


def test_short_section_range():
 n=nav_graph()
 for i in range(12):n.nodes[f'b{i}']['text']='short'
 assert n.bounded_section('S2')==[f'b{i}' for i in range(12)]


def test_b_c_same_base_sources_identical_and_regular_pages_without_gaps(configured):
 c,store,_=configured;base=current_index(c,store,'structured');cat=OriginalCatalog(store,base)
 hits=[next(x for x in cat.chunks.values() if x['kind']=='table')]+[x for x in cat.chunks.values() if x['kind']=='text'][:2]
 r1=StructuralReader(c,store,base);r1.question='fictional revenue'
 r2=StructuralReader(c,store,base);r2.question='fictional revenue'
 a=assemble_restored(r1,hits,base,cat);assemble_restored(r2,hits,base,cat)
 assert r1.public_evidence()==r2.public_evidence()
 assert a['contract']=='BC-source-priority-v3' and len(a['phases'])==6
 assert a['page_fallbacks'] and any(p['reason']=='selected_entrance_bound_source_page' for p in a['page_fallbacks'])
 allowed={(s['doc_version'],s['page_index']) for n in a['anchor_node_ids'] for s in cat.nodes[n]['sources']}
 assert {(p['source']['doc_version'],p['source']['page_index']) for p in a['page_fallbacks']}<=allowed
 assert r1.text_used<=25000 and len(r1.images)<=8
 assert c.query.text_tokens==40000 and r1.config.query.text_tokens==25000


def test_original_coverage_equivalent_summary_refill(configured):
 c,store,_=configured;base=current_index(c,store,'structured');cat=OriginalCatalog(store,base)
 hits=list(cat.chunks.values())[:4];one=hits[0]
 # Original identity resolver is unchanged; synthetic wrapper simulates summary binding.
 summary={**one,'id':'summary','source_chunk_ids':[one['id']],'enrichment_id':'fiction'}
 out,log=select_coverage([one,summary,*hits[1:]],20,cat,base)
 assert log[1]['reason']=='fully_covered'
 assert base_coverage(one,cat,base)==base_coverage(summary,cat,base)


def test_budget_stops_are_explicit_and_no_page_recursion(configured):
 c,store,_=configured;c.query.text_tokens=30;base=current_index(c,store,'structured');cat=OriginalCatalog(store,base)
 r=StructuralReader(c,store,base);r.question='fictional'
 out=assemble_restored(r,list(cat.chunks.values())[:10],base,cat)
 assert r.text_used<=30 and out['unread_ranges']
 assert any(g['unread_node_ranges'] for g in out['resolved_groups'])
 assert len(out['title_matches'])<=4


def test_page_geometry_never_suppresses_partial_node(configured):
 c,store,_=configured;base=current_index(c,store,'structured');cat=OriginalCatalog(store,base)
 r=StructuralReader(c,store,base);node=next(n for n in cat.nodes.values() if n.get('sources') and n['kind']=='text')
 r._read_limit=2;r._text(node['id'],node['text'],node['sources'],0,node['kind'])
 result=read_bound_page(r,node['sources'][0])
 assert result['uncertain_overlap_evidence_ids']


def test_runner_restoration_only_B_C_and_RRF20(configured):
 from financial_rag.runner import run
 c,store,f=configured;c.query.bc_assembly='BC-source-priority-v3';c.retrieval.rrf_k=20
 from test_five_arm_contract import prepare
 prepare(c,store)
 run(c,store,'restoration-fixture',['A','B','C','D','pre-D'],limit=1)
 rows=store.records('restoration-fixture');assert all(r['status']=='ok' for r in rows),[(r['arm'],r['error']) for r in rows]
 for r in rows:
  searches=[t for t in r['trace'] if t.get('tool')=='search']
  if r['arm'] in ['B','C']:
   assert len(searches)==1 and r['model_calls']==1
   assert searches[0]['rrf_k']==20 and 'coverage_selection' in searches[0]
   for entry in searches[0]['coverage_selection']:
    expected=sum(1/(20+entry[k]) for k in ['sparse_rank','dense_rank'] if entry[k] is not None)
    assert entry['rrf_score']==pytest.approx(expected)
   assert r['fixed_assembly']['contract']=='BC-source-priority-v3'
  else:assert all('coverage_selection' not in t for t in searches)
 d=next(r for r in rows if r['arm']=='D');p=next(r for r in rows if r['arm']=='pre-D')
 assert d['initial_snapshot_sha256']==p['initial_snapshot_sha256']
 assert d['initial_expansion']['window_policy']=='D-fixed-eight-v1'
 from financial_rag.evaluation import evaluate
 assert evaluate(c,store,'restoration-fixture')['pending']==0
 assert all(r['fixed_assembly']['contract']=='BC-source-priority-v3' for r in rows if r['arm'] in ('B','C'))


def test_title_only_removed_but_unique_explicit_reference_retained(configured,monkeypatch):
 import financial_rag.bc_restoration as bc
 c,_,_=configured;n=nav_graph();nodes=n.nodes
 # Every extra heading remains a direct child of P, not of another chapter.
 for i in range(6):
  sid=f'title{i}';nodes[sid]={'id':sid,'parse_id':'p','kind':'section_header','parent_id':'P','section_id':sid,'reading_order':100+i*2,'text':f'Credit topic {i}','related_refs':[],'sources':[]}
  nodes[sid+'body']={'id':sid+'body','parse_id':'p','kind':'text','parent_id':sid,'section_id':sid,'reading_order':101+i*2,'text':'Fictional original definition','related_refs':[],'sources':[]}
 nodes['b0']['kind']='table';nodes['b0']['text']='Table 9 fictional values'
 nodes['a11']['text']='See Table 9 for the credit amounts.'
 cat=SimpleNamespace(nodes=nodes,by_ref={},chunks={'anchor':{'node_ids':['a11']}})
 cat.resolve=lambda hit,index:['anchor']
 reader=SimpleNamespace(config=c,tokenizer=n.s.tokenizer,index={'id':'b'},image_limit=8,evidence={},images={},text_used=0,question='Compare credit',remaining=40000)
 def fake_read(reader,structure,ids,**kw):return {'results':[],'unread_ranges':[]}
 monkeypatch.setattr(bc,'read_nodes',fake_read)
 out=bc.assemble_restored(reader,[{'id':'anchor','kind':'text'}],{'id':'b'},cat)
 assert out['title_matches']==[]
 assert not any(x.startswith('title') for x in out['selected_node_ids'])
 assert 'b0' in out['events'][1]['selected_node_ids']
 assert 'outside' not in out['selected_node_ids']
 # Once ambiguous, an original reference cannot choose one target by proximity.
 nodes['b1']['kind']='caption';nodes['b1']['text']='Table 9 another candidate'
 out=bc.assemble_restored(reader,[{'id':'anchor','kind':'text'}],{'id':'b'},cat)
 assert any(g['reason']=='reference_ambiguous' for g in out['structural_gaps'])


def test_rrf_stable_tie_break_and_legacy_path_untouched(configured):
 from financial_rag.retrieval import Search,embed_index
 from financial_rag.api import ModelAPI
 c,store,_=configured;c.retrieval.rrf_k=20;embed_index(c,store,'structured');idx=current_index(c,store,'structured');cat=OriginalCatalog(store,idx)
 search=Search(c,store,idx,ModelAPI(c,store,'fixture-rrf'))
 search.search('fictional assets',coverage_selector=lambda hits,k:select_coverage(hits,k,cat,idx))
 log=search.trace[-1];pairs=[(x['rrf_score'],x['candidate_id']) for x in log['coverage_selection']]
 assert pairs==sorted(pairs,key=lambda x:(-x[0],x[1]))
 search.search('fictional assets');assert 'coverage_selection' not in search.trace[-1]


def test_enriched_entrance_without_summary_same_B_packet_and_summary_full_members(configured):
 from test_five_arm_contract import prepare
 c,store,_=configured;prepare(c,store);base=current_index(c,store,'structured');enriched=current_index(c,store,'enriched');cat=OriginalCatalog(store,base)
 original=next(x for x in cat.chunks.values() if x['kind']=='text')
 equivalent=next(x for x in store.chunks(enriched['id']) if not x.get('enrichment_id') and x.get('source_chunk_ids')==[original['id']])
 left=StructuralReader(c,store,base);right=StructuralReader(c,store,base);left.question=right.question='fictional revenues'
 assemble_restored(left,[original],base,cat);assemble_restored(right,[equivalent],enriched,cat)
 assert left.public_evidence()==right.public_evidence()
 summary=next(x for x in store.chunks(enriched['id']) if x['kind']=='body_summary')
 expected=set(n for cid in summary['source_chunk_ids'] for n in cat.chunks[cid]['node_ids'])
 r=StructuralReader(c,store,base);r.question='fictional revenue';result=assemble_restored(r,[summary],enriched,cat)
 assert expected<=set(result['anchor_node_ids'])
 assert all(e.get('source_key')!='chunk:'+summary['id'] for e in r.public_evidence())
