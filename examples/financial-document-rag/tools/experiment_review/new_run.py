"""Build a new experiment's terminal audit and report from its own raw records.

No model client is instantiated, and no answer, label or score is modified.
All current runs use this single audit implementation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import itertools
import json
import math
import statistics
import zlib
from functools import lru_cache
from pathlib import Path

import pymupdf

from financial_rag.api import usage_summary
from financial_rag.common import RagError, canonical, digest, read_json, read_jsonl, write_json
from financial_rag.config import ARMS, load_config
from financial_rag.contract_report import paired, holm
from financial_rag.context_capacity import Capacity
from financial_rag.evidence_packet import OriginalCatalog
from financial_rag.bc_restoration import select_coverage, select_supplement, Navigation
from financial_rag.structural_expansion import Structure
from financial_rag.tokenization import Tokenizer
from ._context import ImmutableStore


def require(condition, message):
    if not condition:
        raise RagError(message)


def judge_key(cfg, row, label):
    task = {'question': row['query'], 'reference_answer': label.get('answer'),
            'reference_details': label.get('raw_answers'), 'candidate_answer': row['answer']}
    return digest({'stage': 'judge-v1', 'model': cfg.models.judge.identity(), 'task': task})


def wilson(correct, n):
    if not n:
        return None
    z = 1.959963984540054
    center = (correct / n + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(correct/n*(1-correct/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center-half, center+half]


def build_review(plan_path: Path, output: Path | None = None) -> dict:
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    plan = read_json(plan_path)
    require(plan.get('kind') == 'maintained_frozen_experiment', 'Use historical review for archived plans')
    cfg = load_config(Path(plan['config']))
    output = Path(output or root/'review').resolve()
    require(not output.is_relative_to(cfg.work_dir) and
            (output == root/'review' or not output.is_relative_to(cfg.query.source_work_dir)),
            'Review output must not overwrite runtime or source inputs')
    require(len(plan['stages']) == 1, 'Review one frozen stage at a time')
    run_id = plan['stages'][0]['run_id']
    arms = plan['stages'][0].get('arms',list(ARMS))
    ids = set(plan['stages'][0]['query_ids'])
    checks = collections.Counter()
    state = read_json(root/'status.json')
    start = state.get('stages', {}).get(run_id, {}).get('api_start', plan.get('api_start', 0))
    output.mkdir(parents=True, exist_ok=True)
    with ImmutableStore(cfg.work_dir) as runtime, ImmutableStore(cfg.query.source_work_dir) as source:
        for store in (runtime, source):
            require(store.db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Database integrity failed')
        run = runtime.get('runs', run_id)
        require(run is not None, 'Run has not started')
        rows = runtime.records(run_id)
        scores = read_jsonl(cfg.work_dir/'runs'/run_id/'scored.jsonl')
        by = {(r['arm'], r['query_id']): r for r in rows}
        scored = {(r['arm'], r['query_id']): r for r in scores}
        expected = {(a, q) for a in arms for q in ids}
        require(len(rows) == len(scores) == len(expected) and set(by) == set(scored) == expected,
                'Missing or duplicate terminal answers/scores; missing execution is not an exhausted failure')
        meta = runtime.meta('evaluation:'+run_id)
        require(meta is not None, 'Scores do not bind current terminal records')
        labels = {r['query_id']: r for r in read_jsonl(Path(run['dataset']['prepared_dir'])/'labels.jsonl')}
        tasks = {}
        for key, row in by.items():
            score = scored[key]
            require(isinstance(score['judge']['correct'], bool), 'A judge is still pending')
            require(row.get('answer') == score.get('answer') and row['status'] == score['status'], 'Scored answer mismatch')
            require(row['index_id'] == run['indices'][ARMS[row['arm']]], 'Wrong arm index')
            require(1 <= row['attempt'] <= 4 and len(row.get('previous_attempts', [])) == row['attempt']-1,
                    'Attempt history is incomplete')
            if row['status'] != 'ok':
                require(row['status'] != 'not_run' and row['attempt'] == 4 and score['judge']['correct'] is False,
                        'Only exhausted four-attempt failures may enter terminal scores as zero')
                checks['exhausted_terminal_records'] += 1
            else:
                require(isinstance(row['answer'], str) and row['answer'].strip(), 'Successful answer is empty')
                tasks[key] = judge_key(cfg, row, labels[row['query_id']])
        attempts = [a for r in rows for a in [*r.get('previous_attempts', []), r]]
        for checkpoint in sorted((root/'internal-attempts'/run_id).glob('round-*/*.json')):
            if checkpoint.name not in {'records.json','record-hashes.json'}:
                continue
            for prior in read_json(checkpoint):
                if prior.get('status')=='ok':
                    current=by[prior['arm'],prior['query_id']]
                    require(current['status']=='ok' and current['attempt']==prior['attempt'], 'Successful attempt replaced')
                    if 'answer' in prior:
                        require(current['answer']==prior['answer'], 'Successful answer replaced')
                    checks['successful_checkpoint_records_present']+=1
            checks['checkpoints']+=1
        calls = [dict(id=i, context=c, role=role, **json.loads(value)) for i,c,role,value in
                 runtime.db.execute('SELECT id,context,role,payload FROM api_calls ORDER BY id')]
        end = max((c['id'] for c in calls), default=start)
        new = [c for c in calls if start < c['id'] <= end]
        body_hashes = set()
        for key, data in runtime.db.execute('SELECT sha256,data FROM request_payloads'):
            require(isinstance(json.loads(zlib.decompress(data)),dict), 'Request body is not readable JSON')
            body_hashes.add(key)
        require(all(c.get('request_payload_sha256') in body_hashes for c in calls), 'Recorded request lacks body')
        tool_calls = collections.defaultdict(dict)
        routing = collections.Counter()
        image_sends = collections.Counter()
        system_messages = collections.defaultdict(set)
        seen_text = collections.defaultdict(set)
        d_text_total = d_text_repeated = 0
        tokenizer = Tokenizer(cfg.tokenizer)
        max_sent_capacity = 0
        cap = Capacity(cfg)
        for call in new:
            raw = runtime.db.execute('SELECT data FROM request_payloads WHERE sha256=?', (call['request_payload_sha256'],)).fetchone()[0]
            body = json.loads(zlib.decompress(raw))
            routing[(call['role'], body['model'], str(call.get('response_model')))] += 1
            for tc in (call.get('response_message') or {}).get('tool_calls', []):
                tool_calls[call['context']][tc['id']] = tc['function']
            if call['role'] != 'chat':
                continue
            arm = call['context'].split(':')[-3]
            system_messages[arm].add(canonical(body['messages'][0].get('content')))
            if arm == 'D':
                for message in body['messages']:
                    content = message.get('content')
                    parts = content if isinstance(content, list) else [{'type':'text','text':content}] if isinstance(content,str) else []
                    for part in parts:
                        if part.get('type') != 'text':continue
                        text = part['text']; size = tokenizer.count(text); key = digest(text)
                        d_text_total += size
                        if key in seen_text[call['context']]:d_text_repeated += size
                        seen_text[call['context']].add(key)
            images = sum(p.get('type') == 'image_url' for m in body['messages'] if isinstance(m.get('content'), list) for p in m['content'])
            require(images <= 8, 'Too many images in an actual request')
            image_sends[arm] += images
            if arm == 'D':
                count = cap.count(body['messages'], body.get('tools'))
                require(count <= cap.limit, 'D sent request exceeds declared capacity proxy')
                max_sent_capacity = max(max_sent_capacity, count)
        direct_systems = [system_messages[a] for a in ('A','B','C','pre-D') if system_messages[a]]
        require(not direct_systems or all(v == direct_systems[0] for v in direct_systems), 'Direct answer SYSTEM differs across arms')
        current_contexts = {c['context'] for c in new if c['role'] == 'judge'}
        old_contexts = {c['context'] for c in calls if c['id'] <= start and c['role'] == 'judge'}
        unique_tasks = set(tasks.values())
        require(all('judge:'+key in current_contexts | old_contexts for key in unique_tasks), 'Judge has no physical provenance')
        require(current_contexts <= {'judge:'+key for key in unique_tasks}, 'Unattributed judge requests')

        documents = {key: source.get('documents', key) for key, in source.db.execute('SELECT id FROM documents')}
        for doc in documents.values():
            require(Path(doc['pdf_path']).is_file(), 'Original PDF missing')
        @lru_cache(maxsize=8192)
        def original(table, key):
            return source.get(table, key)
        @lru_cache(maxsize=32)
        def page(version, index, kind):
            with pymupdf.open(documents[version]['pdf_path']) as pdf:
                return pdf[index].get_text(kind, sort=True)
        catalog = OriginalCatalog(source, source.get('indices', run['indices']['structured']))
        nav = Navigation(Structure(catalog, Tokenizer(cfg.tokenizer)))
        for row in attempts:
            context = f"run:{run_id}:{row['arm']}:{row['query_id']}:{row.get('attempt', 1)}"
            evidence = row.get('evidence', [])
            for e in evidence:
                for binding in e['sources']:
                    require(binding['doc_version'] in documents and binding['doc_id'] == documents[binding['doc_version']]['doc_id'], 'Invalid source binding')
                    checks['source_bindings'] += 1
                if e['type'] == 'text':
                    key = e['source_key']
                    if key.startswith('chunk:'):
                        obj = original('chunks', key[6:]); require(not obj.get('enrichment_id'), 'Summary used as original evidence')
                        text = obj['text']; require(obj['sources'] == e['sources'], 'Chunk source mismatch')
                    elif key.startswith('raw-page-line:'):
                        _, version, index, block, line = key.split(':')
                        item = page(version, int(index), 'dict')['blocks'][int(block)]['lines'][int(line)]
                        text = ''.join(span['text'] for span in item['spans'])
                        require(e['sources'][0]['bbox'] == list(item['bbox']), 'Raw page geometry mismatch')
                        checks['raw_page_lines'] += 1
                    elif ':page:' in key:
                        version, index = key.split(':page:'); text = page(version, int(index), 'text')
                    else:
                        obj = original('nodes', key); text = obj['text']
                        require(obj['sources'] == e['sources'], 'Node source mismatch')
                    require(text[e['offset']:e['end_offset']] == e['text'], 'Original text changed')
                    checks['original_text_slices'] += 1
                else:
                    binding = e['sources'][0]
                    key = digest([documents[binding['doc_version']]['sha256'], binding['page_index'], cfg.query.image_scale, None])
                    require((source.root/'page_images'/(key+'.png')).read_bytes().startswith(b'\x89PNG'), 'Missing original image')
                    checks['image_bindings'] += 1
            for trace in row.get('trace', []):
                if trace.get('actor') == 'model':
                    tc = tool_calls[context].get(trace.get('tool_call_id'))
                    require(tc and tc['name'] == trace['tool'], 'Tool execution has no model response owner')
                    checks['model_tool_ownership'] += 1
                    response = trace.get('response') or {}
                    if isinstance(response, dict) and trace['tool'] == 'read_parent' and response.get('parent_section_ids'):
                        parents = response['parent_section_ids']
                        for parent in parents:
                            require(parent in catalog.nodes and any(nav.within(anchor,parent) for anchor in response['anchor_node_ids']), 'Parent reading escaped anchor ancestry')
                        allowed = {n for n in catalog.nodes if any(nav.within(n,p) for p in parents)}
                        related, _ = nav.s.related(list(allowed))
                        require(set(response.get('selected_node_ids', [])) <= allowed | set(related), 'Parent reading escaped subtree and explicit relations')
                        checks['D_parent_scope'] += 1
                    if isinstance(response, dict) and trace['tool'] == 'read_source_pages' and response.get('bound_pages'):
                        anchors=[]
                        for sid in trace['arguments']['source_ids']:
                            anchors.extend(nav.s.targets(chunk_id=sid) if sid in catalog.chunks else nav.s.targets(node_id=sid))
                        allowed={(s['doc_version'],s['page_index']) for nid in anchors for s in catalog.nodes[nid].get('sources',[])}
                        require({(s['doc_version'],s['page_index']) for s in response['bound_pages']} <= allowed, 'D read unbound source page')
                        checks['D_bound_pages'] += 1
                if trace.get('tool') != 'search':
                    continue
                require(trace['index_id'] == run['indices'][ARMS[row['arm']]], 'Search used another arm index')
                fused = collections.defaultdict(float)
                for lane in ('sparse', 'dense'):
                    require(len(trace[lane]) <= 50, 'Too many retrieval candidates')
                    for rank, key in enumerate(trace[lane], 1):
                        fused[key] += 1/(20+rank)
                raw = [{**original('chunks', key), 'score': score} for key,score in sorted(fused.items(), key=lambda x:(-x[1],x[0]))]
                if row['arm'] in ('B','C'):
                    primary=[]
                    if row['arm']=='C':
                        base=trace.get('original_search')
                        require(base and base['index_id']==run['indices']['structured'] and base['query']==trace['query'],
                                'C must preserve the same-query B-index retrieval')
                        base_fused=collections.defaultdict(float)
                        for lane in ('sparse','dense'):
                            require(len(base[lane])<=50,'Too many original candidates')
                            for rank,key in enumerate(base[lane],1):base_fused[key]+=1/(20+rank)
                        base_raw=[{**original('chunks',key),'score':score} for key,score in sorted(base_fused.items(),key=lambda x:(-x[1],x[0]))]
                        primary,base_log=select_coverage(base_raw,20,catalog,catalog.index)
                        for entry in base_log:
                            for lane in ('sparse','dense'):
                                ranking=base[lane];entry[lane+'_rank']=ranking.index(entry['candidate_id'])+1 if entry['candidate_id'] in ranking else None
                        require(base_log==base['coverage_selection'],'C original coverage mismatch')
                        require(base['ranked_chunks']==[{'id':h['id'],'score':h['score']} for h in primary],'C original ranking mismatch')
                        require(trace['protected_original_ids']==[h['id'] for h in primary],'C discarded original anchors')
                        extra,log=select_supplement(raw,cfg.retrieval.summary_supplement_k,catalog,source.get('indices',trace['index_id']),primary)
                        require(trace['summary_supplement_ids']==[h['id'] for h in extra],'C summary selection mismatch')
                        selected=primary+extra
                        checks['C_protected_original_searches']+=1
                    else:
                        selected, log = select_coverage(raw,20,catalog,source.get('indices',trace['index_id']))
                    for entry in log:
                        for lane in ('sparse', 'dense'):
                            ranking = trace[lane]
                            entry[lane+'_rank'] = ranking.index(entry['candidate_id'])+1 if entry['candidate_id'] in ranking else None
                    require(log == trace['coverage_selection'], 'BC coverage/refill mismatch')
                else:
                    groups = {}
                    for hit in raw:
                        assets = tuple(sorted(hit.get('asset_ids', [])))
                        key = ('asset',assets) if assets else ('source_set',tuple(hit.get('source_chunk_ids',[hit['id']])))
                        groups.setdefault(key,hit)
                    limit=cfg.retrieval.flat_final_k if row['arm']=='A' else 20
                    selected = list(groups.values())[:limit]
                require(trace['ranked_chunks'] == [{'id':h['id'],'score':h['score']} for h in selected], 'RRF20 ranking mismatch')
                checks['RRF20_searches'] += 1
            if row['arm'] in ('B','C') and row.get('fixed_assembly'):
                assembly = row['fixed_assembly']
                require(assembly['contract']==cfg.query.bc_assembly,'BC policy identity changed')
                require(not assembly['title_matches'] and row['text_tokens_read'] <= 25000 and row['page_images_read'] <= 8, 'BC strategy limit mismatch')
                require(len(assembly['events']) == 6 and assembly['events'][-1]['new_tokens'] <= 4000, 'BC stage contract mismatch')
                require([x['phase'] for x in assembly['phases']]==[
                    'anchors_and_related','explicit_relations','local_structure','parent_overview',
                    'associated_sections_and_cross_heading','bound_original_pages'],'BC priority order changed')
                for event in assembly['events']:
                    for rel in event.get('relations', []):
                        if rel['relation'] == 'parent_direct_overview':
                            require(rel['parent'] == nav.parent_of_anchor(rel['anchor']) and rel['nodes'] == nav.direct(rel['parent']), 'Parent overview mismatch')
                        if rel['relation'] == 'cross_local_heading_eight':
                            require((rel['nodes'],rel['parent']) == nav.adjacent(rel['anchor']), 'Cross-heading range mismatch')
                checks['BC_assemblies'] += 1
            if row['arm'] == 'D':
                require(row.get('model_calls',0) <= 25, 'D logical call limit exceeded')
                if row.get('model_calls') == 25:
                    last = [t for t in row['trace'] if t.get('tool') == 'model_call'][-1]
                    require(last['phase'] == 'forced_final' and not last['tools_enabled'], 'Call25 must disable tools')
        for q in ids:
            if 'B' in arms and 'C' in arms and by['B',q].get('fixed_assembly') and by['C',q].get('fixed_assembly'):
                b,c=by['B',q],by['C',q]
                require(b['fixed_assembly']['protected_original_node_ids']==c['fixed_assembly']['protected_original_node_ids'],
                        'C original anchor set differs from B')
                count=b['fixed_assembly']['phases'][0]['evidence_count']
                require(b['evidence'][:count]==c['evidence'][:count],'C changed or displaced B base evidence')
                checks['BC_primary_evidence_prefix_matches']+=1
        for q in (ids if 'D' in arms else []):
            saved = runtime.cached(digest(['initial-evidence-v1',run_id,q]))
            if saved:
                require(isinstance(saved.get('packet'),dict), 'Initial snapshot missing')
                for arm in ('D','pre-D'):
                    if by[arm,q]['status'] == 'ok':
                        require(by[arm,q]['query'] == saved['packet']['question'], 'Initial snapshot question mismatch')
                if by['pre-D',q]['status'] == 'ok':
                    require(by['pre-D',q]['evidence'] == [{k:v for k,v in e.items() if k!='data_url'} for e in saved['packet']['evidence'].values()], 'pre-D contains expanded or changed evidence')
                checks['shared_snapshots'] += 1
            else:
                require(by['D',q]['status'] != 'ok' and by['pre-D',q]['status'] != 'ok', 'Valid D/pre-D requires initial snapshot')

        truth = {a:{q:scored[a,q]['judge']['correct'] for q in ids} for a in arms}
        stats = {a:{'n':len(ids),'correct':sum(truth[a].values()),'accuracy':sum(truth[a].values())/len(ids),
                    'wilson95':wilson(sum(truth[a].values()),len(ids)),
                    'mean_original_tokens_estimated':statistics.mean(by[a,q].get('text_tokens_read',0) for q in ids)} for a in arms}
        pairs = {a+'->'+b:paired(truth[a],truth[b]) for a,b in itertools.combinations([a for a in 'ABCD' if a in arms],2)}; holm(pairs)
        diagnostic = {a+'->'+b:paired(truth[a],truth[b]) for a,b in [('B','pre-D'),('pre-D','D')] if a in arms and b in arms}; holm(diagnostic)
        phases = {}
        for arm in ('B','C'):
            values = collections.defaultdict(list)
            for q in ids:
                previous = 0
                for phase in (by[arm,q].get('fixed_assembly') or {}).get('phases',[]):
                    values[phase['phase']].append(phase['text_tokens']-previous); previous=phase['text_tokens']
            phases[arm] = {key:{'mean_added_tokens_estimated':statistics.mean(v),'records_observed':len(v)} for key,v in values.items()}
        strata = {}
        for item in plan.get('development_files', []):
            if Path(item['path']).name != 'development-overlap.json':continue
            overlap = read_json(Path(item['path']))
            for name,key in [('current_development','current49_development_ids'),('all_development','earlier54_and_current_union_ids')]:
                dev = ids & set(overlap[key])
                for label,group in [(name,dev),('outside_'+name,ids-dev)]:
                    strata[label] = {a:{'n':len(group),'correct':sum(truth[a][q] for q in group)} for a in arms}
        d_reading = []
        for q in (sorted(ids) if 'D' in arms else []):
            row = by['D',q]
            saved = runtime.cached(digest(['initial-evidence-v1',run_id,q]))
            if row['status'] != 'ok' or not saved:continue
            anchor = saved['packet']['text_used'] if 'text_used' in saved['packet'] else by['pre-D',q]['text_tokens_read']
            auto = row['initial_expansion']['new_original_tokens_estimated']
            extra = row['text_tokens_read']-anchor-auto
            require(extra >= 0, 'D expansion accounting is inconsistent')
            d_reading.append({'query_id':q,'initial_anchor_tokens_estimated':anchor,'automatic_new_tokens_estimated':auto,
                              'model_new_tokens_estimated':extra,'final_tokens_estimated':row['text_tokens_read'],
                              'model_tools':[{'tool':t['tool'],'response':t.get('response'),'new_evidence':t.get('new_evidence')} for t in row['trace'] if t.get('actor')=='model']})
        source_calls = [dict(role=role,**json.loads(payload)) for role,payload in source.db.execute('SELECT role,payload FROM api_calls')]
        new_used = {c['request_payload_sha256'] for c in calls}
        result = {'kind':'new_run_terminal_review','run_id':run_id,'scope':plan['scope'],'arms':arms,'records':len(rows),'questions_per_arm':len(ids),
                  'retrieval_contract':{'A_final_k':cfg.retrieval.flat_final_k,'B_original_final_k':cfg.retrieval.final_k,
                                        'C_summary_supplement_k':cfg.retrieval.summary_supplement_k,'BC_policy':cfg.query.bc_assembly},
                  'terminal_valid_answers':sum(r['status']=='ok' for r in rows),'stats':stats,'main_paired':pairs,'diagnostic_paired':diagnostic,
                  'development_strata':strata,'checks':dict(checks),'attempts':len(attempts),'api_start':start,'api_end':end,
                  'new_physical_requests':len(new),'new_usage':usage_summary(new),
                  'generation_usage_by_arm':{a:usage_summary([c for c in new if c['context'].startswith(f'run:{run_id}:{a}:')]) for a in arms},
                  'shared_setup_usage':usage_summary([c for c in new if c['context'].startswith('setup:')]),
                  'source_historical_usage_not_new_run':usage_summary(source_calls),
                  'judge':{'scored_records':len(tasks),'unique_tasks':len(unique_tasks),
                           'duplicate_task_cache_reuse':len(tasks)-len(unique_tasks),
                           'tasks_reused_from_before_boundary':sum('judge:'+key in old_contexts and 'judge:'+key not in current_contexts for key in unique_tasks),
                           'physical_requests':sum(c['role']=='judge' for c in new)},
                  'unknown_unassociated_request_bodies':sorted(body_hashes-new_used),
                  'actual_routing':[{'role':k[0],'requested_model':k[1],'returned_model':k[2],'requests':v} for k,v in routing.items()],
                  'http_statuses':{role:dict(collections.Counter(str(c.get('status_code')) for c in new if c['role']==role)) for role in ('chat','embedding','judge')},
                  'physical_inline_image_occurrences':dict(image_sends),'D_max_sent_capacity_proxy':max_sent_capacity,
                  'D_reading':d_reading,
                  'D_repeated_text_proxy':{'total':d_text_total,'repeated_identical_parts':d_text_repeated,
                                          'definition':'Within-attempt identical message text including system/question; not exact provider billing or useless-token attribution'},
                  'capacity_diagnostics':{'rejections':sum(t.get('tool')=='capacity_rejection' for r in attempts for t in r.get('trace',[])),
                                          'image_omission_events':sum(bool(t.get('image_pointers')) for r in attempts for t in r.get('trace',[])),
                                          'compaction_events':sum(bool(t.get('compacted_message_indices')) for r in attempts for t in r.get('trace',[]))},
                  'BC_phase_reading':phases,'invalid_citation_records':sum(bool(r.get('citation_check',{}).get('invalid')) for r in rows),
                  'valid_answer_empty_citation_records':sum(r['status']=='ok' and not r.get('citations') for r in rows),
                  'zero_provider_calls_for_review':True,'currency_cost_verified':False,
                  'hash_verification':False}
        cases = []
        for compare,data in {**pairs,**diagnostic}.items():
            left,right = compare.split('->')
            for outcome in ('gains','losses'):
                for q in data[outcome]:
                    cases.append({'comparison':compare,'outcome':outcome,'query_id':q,'question':by[left,q]['query'],'original_reference':labels[q],
                                  'left':{'answer':by[left,q].get('answer'),'judge':scored[left,q]['judge'],'evidence':by[left,q].get('evidence',[])},
                                  'right':{'answer':by[right,q].get('answer'),'judge':scored[right,q]['judge'],'evidence':by[right,q].get('evidence',[])}})
    write_json(output/'analysis.json', result)
    write_json(output/'cases.json', cases)
    write_json(output/'audit.json', {'checks':dict(checks),'records':len(rows),'attempts':len(attempts),
                                    'all_terminal_records_scored':True,'only_exhausted_failures_score_zero':True})
    with (output/'terminal-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','query_id','correct','answer','attempts','original_tokens_estimated'])
        writer.writeheader()
        for row in scores:writer.writerow({'arm':row['arm'],'query_id':row['query_id'],'correct':row['judge']['correct'],
                                         'answer':row.get('answer'),'attempts':row['attempt'],'original_tokens_estimated':row.get('text_tokens_read',0)})
    render_report(result,output)
    write_json(output/'artifacts.json',{p.name:{'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name != 'artifacts.json'})
    return result


def render_report(result, output):
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验终态报告</title><style>body{font:16px/1.7 system-ui,Microsoft YaHei;max-width:1100px;margin:40px auto;color:#18293d}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:8px}code{overflow-wrap:anywhere}</style><h1>实验终态报告</h1>',
           '<p>运行：'+html.escape(result['run_id'])+'。本报告由本轮原始材料生成，生成过程不调用模型或修改评分。</p>',
           '<h2>统一终态成绩</h2>',table(['组别','正确/题数','正确率','平均原文token估计'],[[a,f"{s['correct']}/{s['n']}",f"{s['accuracy']:.2%}",round(s['mean_original_tokens_estimated'])] for a,s in result['stats'].items()]),
           '<p>补跑耗尽按错计，分母固定；不分别发布各轮补跑成绩。当前小样用于流程验证，不是独立性能结论。</p>' if result['questions_per_arm']<=5 else '<p>补跑耗尽按错计，分母固定；不分别发布各轮补跑成绩。题集曾参与开发与历史观察，不称盲测。</p>',
           '<h2>配对得失</h2>',table(['比较','改善','退化','净增','精确p','Holm p'],[[name,len(p['gains']),len(p['losses']),p['net'],round(p['exact_two_sided_p'],6),round(p['holm_p'],6)] for name,p in {**result['main_paired'],**result['diagnostic_paired']}.items()]),
           '<p>主比较与存在的诊断比较分别做Holm校正。B→C检验保护原始入口后的摘要补充。若包含D/pre-D，B→pre-D同时改变查询和初始装配，pre-D→D包含固定首扩展与后续迭代。未运行的组不填入历史答案。</p>',
           '<h2>全部物理调用与费用边界</h2>',table(['角色','物理请求','已报总token','用量未知请求'],[[role,u['requests'],u['total_tokens_reported'],u['usage_missing_requests']] for role,u in result['new_usage'].items()]),
           f"<p>API边界({result['api_start']}, {result['api_end']}]，共{result['new_physical_requests']}次物理请求，包含重试和全部尝试。评分任务按问题、参考、候选与judge身份关联；新增和历史缓存不重复计费。{len(result['unknown_unassociated_request_bodies'])}个未关联账本的body提交/用量未知，不能算零。无实际账单，不虚构货币费用。</p>",
           '<h2>模型与原图实际路由</h2>',table(['角色','请求模型','返回别名','请求数'],[[r['role'],r['requested_model'],r['returned_model'],r['requests']] for r in result['actual_routing']]),
           '<p>响应别名不能证明上游权重。图像由chat inline发送，不代表存在独立vision推理。各组物理图像发送次数（含重发）：'+html.escape(str(result['physical_inline_image_occurrences']))+'</p>',
           '<h2>来源、读取与限制</h2><p>当前参与组：'+html.escape(', '.join(result['arms']))+'。策略：'+html.escape(str(result['retrieval_contract']))+'。A 组本轮选择 10 个入口，历史批次选择 20 个入口，读取范围不同，应按批次报告；B/C 超过本轮 A 不能单独证明优于历史批次的 A。B/C共同25k/8图，明确关系前移；C同一查询执行原始与摘要两次索引查找，复用查询向量，一次回答。来源与检索核查记录资料来源和检索过程；内容是否语义正确、模型是否实际看到已加载的原图，仍需分别判断。D规则仅在本轮包含D时适用。</p>',
           table(['诊断','数量'],[['原文切片核对',result['checks'].get('original_text_slices',0)],['RRF20搜索核对',result['checks'].get('RRF20_searches',0)],['模型工具归属',result['checks'].get('model_tool_ownership',0)],['无效引用记录',result['invalid_citation_records']],['有效答案无正式引用',result['valid_answer_empty_citation_records']]]),
           '<p>逐阶段读取量、开发题分层、Wilson区间、缓存归属和所有配对题号见analysis.json。cases.json保留得失两侧答案、原judge与原文；标签争议需人工结合原文说明，不自动改标签。</p>',
           '<p><a href="analysis.json">完整统计</a> · <a href="audit.json">来源与执行核查</a> · <a href="cases.json">原文得失案例</a> · <a href="terminal-results.csv">逐题结果</a> · <a href="artifacts.json">文件清单</a></p></html>']
    (output/'REPORT.zh-CN.html').write_text(''.join(parts),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=build_review(args.plan,args.output)
    print(json.dumps({'run_id':result['run_id'],'records':result['records'],'zero_provider_calls_for_review':True},ensure_ascii=False))


if __name__ == '__main__':
    main()
