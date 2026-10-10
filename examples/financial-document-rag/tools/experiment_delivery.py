"""Combine newly audited ABC with unchanged historical D/pre-D; never generate or judge."""
import csv
import html
import itertools
import json
import sqlite3
import statistics
import collections
from pathlib import Path

from financial_rag.api import usage_summary
from financial_rag.common import read_json, read_jsonl, write_json
from financial_rag.config import load_config
from financial_rag.contract_report import paired, holm
from experiment_reuse import unpack, verify


def build_delivery(plan_path):
    plan_path=Path(plan_path);plan=read_json(plan_path);root=plan_path.parent
    cfg=load_config(Path(plan['config']));stage=plan['stages'][0]
    reused=Path(plan['reuse_material']['path'])
    reuse_check=verify(reused)
    analysis=read_json(root/'review/analysis.json');audit=read_json(root/'review/audit.json')
    assert audit['all_terminal_records_scored'] and audit['only_exhausted_failures_score_zero']
    for name in ('queries.jsonl','labels.jsonl'):
        assert read_jsonl(cfg.work_dir/'prepared'/name)==read_jsonl(reused/name), 'Reused cohort or labels differ'
    current=read_jsonl(cfg.work_dir/'runs'/stage['run_id']/'scored.jsonl')
    ids=set(stage['query_ids']);assert stage['arms']==['A','B','C']
    db=sqlite3.connect((reused/'reuse.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    retained={(a,q):unpack(data) for a,q,data in db.execute('SELECT arm,query_id,data FROM scores')}
    historical_calls=[{'id':i,'context':c,'role':r,**json.loads(p)} for i,c,r,p in db.execute('SELECT * FROM api_calls ORDER BY id')]
    routing=collections.Counter();sent_images=collections.Counter()
    for call in historical_calls:
        body=unpack(db.execute('SELECT data FROM request_payloads WHERE sha256=?',(call['request_payload_sha256'],)).fetchone()[0])
        routing[call['role'],body['model'],str(call.get('response_model'))]+=1
        if call['role']=='chat':
            arm=call['context'].split(':')[-3]
            for message in body.get('messages',[]):
                content=message.get('content')
                if isinstance(content,list):sent_images[arm]+=sum(part.get('type')=='image_url' for part in content)
    db.close()
    rows=current+list(retained.values());arms=['A','B','C','pre-D','D']
    assert len(rows)==len(ids)*5 and {(r['arm'],r['query_id']) for r in rows}=={(a,q) for a in arms for q in ids}
    assert all(isinstance(r['judge']['correct'],bool) for r in rows)
    truth={a:{r['query_id']:r['judge']['correct'] for r in rows if r['arm']==a} for a in arms}
    pairs={a+'->'+b:paired(truth[a],truth[b]) for a,b in itertools.combinations(arms,2)};holm(pairs)
    stats={a:{'correct':sum(truth[a].values()),'n':len(ids),'accuracy':sum(truth[a].values())/len(ids),
              'origin':'new' if a in stage['arms'] else 'reused',
              'run_id':stage['run_id'] if a in stage['arms'] else read_json(reused/'manifest.json')['source_run_id'],
              'mean_original_tokens_estimated':statistics.mean(r['text_tokens_read'] for r in rows if r['arm']==a),
              'mean_loaded_images':statistics.mean(r['page_images_read'] for r in rows if r['arm']==a)} for a in arms}
    assert stats['pre-D']['correct']==reuse_check['correct']['pre-D'] and stats['D']['correct']==reuse_check['correct']['D']
    result={'total_records':len(rows),'new_records':len(current),'reused_records':len(retained),
            'stats':stats,'paired':pairs,'ABC_audit':audit,'reuse_verification':reuse_check,
            'new_usage':analysis['new_usage'],'reused_historical_usage_not_new':usage_summary(historical_calls),
            'reuse_material':plan['reuse_material'],'historical_unknown_usage':read_json(reused/'manifest.json')['unknown_usage'],
            'analysis':analysis,'new_provider_calls_for_delivery':0,'currency_bill_available':False}
    historical_attempts=[x for row in retained.values() for x in [*row.get('previous_attempts',[]),row]]
    result['reused_diagnostics']={
        'actual_routing':[{'role':k[0],'requested_model':k[1],'returned_model':k[2],'requests':v} for k,v in routing.items()],
        'physical_inline_image_occurrences':dict(sent_images),
        'invalid_citation_records':sum(bool(r.get('citation_check',{}).get('invalid')) for r in retained.values()),
        'valid_answer_empty_citation_records':sum(r['status']=='ok' and not r.get('citations') for r in retained.values()),
        'image_omission_events':sum(bool(t.get('image_pointers')) for r in historical_attempts for t in r.get('trace',[])),
        'compaction_events':sum(bool(t.get('compacted_message_indices')) for r in historical_attempts for t in r.get('trace',[])),
        'query_vectors_and_provenance_retained':reuse_check['query_vectors']}
    output=root/'delivery';output.mkdir(exist_ok=True)
    write_json(output/'analysis.json',result)
    with (output/'per-question.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['arm','query_id','source_run_id','origin','correct','status','attempt','original_tokens_estimated','loaded_images','judge_reason']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            is_new=r['arm'] in stage['arms']
            writer.writerow(dict(arm=r['arm'],query_id=r['query_id'],source_run_id=r['run_id'],origin='new' if is_new else 'reused',
                correct=r['judge']['correct'],status=r['status'],attempt=r['attempt'],original_tokens_estimated=r['text_tokens_read'],
                loaded_images=r['page_images_read'],judge_reason=r['judge'].get('reason')))
    def table(head,values):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in values)+'</table>'
    text=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五组实验最终汇总</title>',
          '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#202b36}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd5de;padding:8px;text-align:left}pre{white-space:pre-wrap}</style>',
          '<h1>五组实验最终汇总</h1><p>888 条新 A/B/C＋592 条复用 pre-D/D，共 1,480 条。执行完成、来源核查和答案正确率分别判断。各组分母固定 296；失败耗尽按错记。</p>',
          table(['组','来源批次','正确/总数','正确率','平均原文估计 token','平均加载图片'],[[a,stats[a]['run_id']+' / '+stats[a]['origin'],f"{stats[a]['correct']}/{stats[a]['n']}",f"{stats[a]['accuracy']:.2%}",round(stats[a]['mean_original_tokens_estimated']),round(stats[a]['mean_loaded_images'],2)] for a in arms]),
          '<h2>方案与解释边界</h2><p>五组统一命名为 A、B、C、pre-D、D。A 最终选择 10 个平面原文入口；B 最多选择 20 个原始入口。C 保留 B 原始入口并补最多 5 个新增覆盖摘要入口；B/C 共用六阶段装配和 25k/8 图上限，不设置跨小节保留额度。C 是同查询两次索引查找、一次回答，复用查询向量。摘要本身不是证据。</p>',
          '<p>D/pre-D 保持原 run_id、记录、评分和请求；本轮没有重新生成或调用 judge。D 的 193 题仍为原四次尝试耗尽零分。当前代码提供相同 D 策略，但删除旧源码后不宣称可完整重建历史代码环境。</p>',
          '<h2>配对得失与不确定性</h2><p>改善/退化按右组相对左组计算。ABC 内部比较属于本轮；涉及 D/pre-D 与 ABC 的比较跨运行批次，存在生成及评分时间差异。精确配对检验和 Holm 校正不消除开发题选择或文档相关性。</p>',
          table(['比较','改善','退化','净增','精确 p','Holm p'],[[k,len(v['gains']),len(v['losses']),v['net'],round(v['exact_two_sided_p'],6),round(v['holm_p'],6)] for k,v in pairs.items()]),
          '<h2>费用与请求</h2><p>以下两张账本分别统计，本次费用不再计入历史复用调用。实际货币账单不可见，token 估计和提供商已报用量不能冒充实际费用。</p>',
          '<h3>本次新增</h3>',table(['角色','物理请求','已报 token','用量未知请求'],[[role,u['requests'],u['total_tokens_reported'],u['usage_missing_requests']] for role,u in result['new_usage'].items()]),
          '<h3>D/pre-D 历史投入（关联请求去重）</h3>',table(['角色','物理请求','已报 token','用量未知请求'],[[role,u['requests'],u['total_tokens_reported'],u['usage_missing_requests']] for role,u in result['reused_historical_usage_not_new'].items()]),
          '<p>原全量主动暂停时存在在途请求提交/用量未知；账本没有孤立请求体也不能将此开销记零。源库摘要构建属于更早历史投入，详见本轮核查报告，不能与新生成账本重复相加。</p>',
          '<h2>来源、图片和案例</h2><p><a href="../review/REPORT.zh-CN.html">查看 ABC 原文来源核查、分阶段读取量、真实发送/省略图片、得失案例与模型路由</a>；逐题答案及原评分理由保存在运行库，全部组别索引见 <a href="per-question.csv">逐题 CSV</a>。来源绑定及加载图片不等于模型实际看到全部原图，也不证明语义正确。</p>',
          '<h3>复用批次的图像与引用诊断</h3><pre>'+html.escape(json.dumps(result['reused_diagnostics'],ensure_ascii=False,indent=2))+'</pre>',
          '<h2>证据限制</h2><p>所有题目历史结果均已被观察，49 道开发题及历史 76 题开发范围须一并披露，本轮不称盲测。原标签与评分规则保持不变，争议单列而不改分。B 到 pre-D 包含改写和初始装配差异；pre-D 到 D 包含固定首扩展及后续主动读取，不能归因单一因素。返回模型别名不能独立证明上游权重。A 组本轮选择 10 个入口，历史批次选择 20 个入口。B/C 高于本轮 A 不单独证明 B/C 自身提高或超过历史批次的 A。</p></html>']
    rendered='\n'.join(text)
    manual=output/'manual-review.html'
    if manual.is_file():
        rendered=rendered.replace('</html>',manual.read_text(encoding='utf-8')+'</html>')
    (output/'REPORT.zh-CN.html').write_text(rendered,encoding='utf-8')
    write_json(output/'artifacts.json',{p.name:{'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name not in {'artifacts.json','SHA256SUMS.json'}})
    write_json(root.parents[1]/'current-delivery.json',{'plan':str(plan_path.resolve()),'report':str((output/'REPORT.zh-CN.html').resolve()),
        'source_runs':{a:v['run_id'] for a,v in stats.items()},'new_records':len(current),'reused_records':len(retained),'hash_verification':False})
    return result
