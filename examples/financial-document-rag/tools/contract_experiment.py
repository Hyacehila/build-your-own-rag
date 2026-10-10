"""Frozen five-arm pilot/full driver; three uniform deferred recovery rounds."""
import argparse,json,os
from pathlib import Path
from datetime import datetime,timezone
from dotenv import load_dotenv
from financial_rag.common import read_json,write_json,read_jsonl,digest
from financial_rag.config import load_config,ARMS
from financial_rag.runner import run
from financial_rag.storage import Store
from financial_rag.evaluation import report
from financial_rag.context_capacity import Capacity


def main(arm=None, argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    snapshots=ap.add_mutually_exclusive_group()
    snapshots.add_argument('--snapshot-plan',type=Path,help='Existing D snapshot run for standalone pre-D')
    snapshots.add_argument('--snapshot-material',type=Path,help='Verified retained D material for standalone pre-D')
    mode=ap.add_mutually_exclusive_group();mode.add_argument('--status',action='store_true')
    mode.add_argument('--check',action='store_true',help='Validate a new frozen plan without provider requests')
    args=ap.parse_args(argv);p=read_json(args.plan);root=args.plan.parent
    state_file=root/'status.json';state=read_json(state_file) if state_file.exists() else {'stages':{}}
    if args.status:print(json.dumps(state,ensure_ascii=False,indent=2));return
    if (p.get('scope') == 'full' or any(len(stage['query_ids']) > 49 for stage in p['stages'])) and not p.get('full_run_authorized', False) and not args.check:
        raise RuntimeError('Full experiment requires a new explicit user decision; this pilot cannot auto-start full.')
    load_dotenv(Path(p['project_root'])/'.env')
    from experiment_setup import check_plan
    checked=check_plan(args.plan,require_credentials=not args.check) if args.check or p.get('kind')=='maintained_frozen_experiment' else None
    if args.check:print(json.dumps(checked,ensure_ascii=False,indent=2));return
    cfg=load_config(Path(p['config']))
    def verify():
        check_plan(args.plan)
    verify();Capacity(cfg)
    snapshot_store=None;snapshot_run_id=None
    if arm:
        if arm not in p['stages'][0].get('arms',list(ARMS)):raise RuntimeError('Arm is outside the frozen plan')
        if arm=='pre-D':
            if not args.snapshot_plan and not args.snapshot_material:raise RuntimeError('pre-D requires --snapshot-plan or --snapshot-material; no D run is started implicitly')
            if args.snapshot_material:
                from experiment_reuse import SnapshotMaterial
                snapshot_store=SnapshotMaterial(args.snapshot_material)
                snapshot_run_id=snapshot_store.manifest['source_run_id']
            else:
                sp=read_json(args.snapshot_plan)
                import tomllib
                historical=tomllib.loads(Path(sp['config']).read_text(encoding='utf-8'))
                snapshot_store=Store((Path(sp['config']).parent/historical['work_dir']).resolve(),read_only=True)
                snapshot_run_id=sp['stages'][0]['run_id']
            # Check the entire cohort before constructing any provider client.
            from financial_rag.dataset import dataset
            with Store(cfg.work_dir,read_only=True) as rs:
                questions={q['query_id']:q['query'] for q in read_jsonl(Path(dataset(rs)['prepared_dir'])/'queries.jsonl')}
            for q in p['stages'][0]['query_ids']:
                v=snapshot_store.cached(digest(['initial-evidence-v1',snapshot_run_id,q]))
                if not v or v['packet']['question']!=questions[q]:
                    raise RuntimeError('Complete matching D snapshots are required before any API request')
    lock=root/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    def save(phase):
        state.update(phase=phase,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_file,state)
    try:
        for stage in p['stages']:
            run_id=stage['run_id'];stage_state=state['stages'].setdefault(run_id,{'completed_round':-1})
            st=stage_state.setdefault('individual_arms',{}).setdefault(arm,{'completed_round':-1}) if arm else stage_state
            identity_arms=stage.get('arms',list(ARMS));arms=[arm] if arm else identity_arms
            if st.get('complete'):continue
            if 'api_start' not in st:
                with Store(cfg.work_dir) as store:
                    st['api_start']=stage_state.setdefault('api_start',p.get('api_start',0))
                save('stage_initialized')
            for round_number in ([] if st.get('generation_complete') else range(st['completed_round']+1,4)):
                verify();save('generation' if round_number==0 else 'uniform_recovery')
                with Store(cfg.work_dir) as store:
                    before={ (r['arm'],r['query_id']):r for r in store.records(run_id) if r['status']=='ok'}
                    run(cfg,store,run_id,arms,query_ids=stage['query_ids'],
                        retry_errors=round_number>0,recovery_round=round_number,identity_arms=identity_arms,
                        snapshot_store=snapshot_store,snapshot_run_id=snapshot_run_id)
                    rows=[r for r in store.records(run_id) if r['arm'] in arms]
                    assert len(rows)==len(stage['query_ids'])*len(arms)
                    assert all(r==before[r['arm'],r['query_id']] for r in rows if (r['arm'],r['query_id']) in before)
                    checkpoint=root/'internal-attempts'/run_id/f'round-{round_number}{"-"+arm if arm else ""}'
                    checkpoint.mkdir(parents=True,exist_ok=True)
                    write_json(checkpoint/'records.json',[{'arm':r['arm'],'query_id':r['query_id'],'attempt':r['attempt'],'status':r['status'],'answer':r.get('answer')} for r in rows])
                    # SQLite retains every attempt; successful records are compared directly before/after recovery.
                    remaining=sum(r['status']!='ok' for r in rows)
                st['completed_round']=round_number;save('round_complete')
                if not remaining:break
            st['generation_complete']=True
            if arm:
                st['complete']=True;save('arm_generation_complete');continue
            verify();save('final_evaluation')
            from experiment_score import score
            scoring=score(args.plan)
            if scoring['pending']:raise RuntimeError('Evaluation service failed; preserve state and resume cached evaluation')
            with Store(cfg.work_dir) as store:
                report(cfg,store,run_id)
            # Closed runtime: the final review reads a consistent SQLite view and
            # writes only derived reports, never answers, scores or provider calls.
            from experiment_review.new_run import build_review
            build_review(args.plan, root/'review')
            if p.get('reuse_material'):
                from experiment_delivery import build_delivery
                build_delivery(args.plan)
            st['complete']=True;save('stage_complete')
        if arm:save('arm_generation_complete');return
        save('pilot_complete_awaiting_user_decision' if p.get('await_user_full_decision') else 'generation_and_final_scores_complete')
    except Exception as e:
        state['error']={'type':type(e).__name__,'message':str(e)};save('stopped_for_review');raise
    finally:
        lock.unlink(missing_ok=True)
        if snapshot_store:snapshot_store.close()


if __name__=='__main__':main()
