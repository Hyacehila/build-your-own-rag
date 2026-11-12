"""The shared scoring entrypoint: generated arms only, resumable judge cache."""
import argparse
import json
from pathlib import Path

from financial_rag.common import RagError, read_json
from financial_rag.config import ARMS, load_config
from financial_rag.evaluation import evaluate
from financial_rag.storage import Store


def score(plan_path):
    plan = read_json(plan_path)
    cfg = load_config(Path(plan["config"]))
    stage = plan["stages"][0]
    with Store(cfg.work_dir) as store:
        run=store.get('runs',stage['run_id'])
        arms=stage.get('arms',list(ARMS))
        if not run or run['arms']!=arms:
            raise RagError('Scoring scope does not match frozen generated arms')
        rows=store.records(stage['run_id'])
        expected={(a,q) for a in arms for q in stage['query_ids']}
        if {(r['arm'],r['query_id']) for r in rows}!=expected:
            raise RagError('Missing execution cannot be scored as failure')
        if any(r['status']!='ok' and r.get('attempt',0)!=4 for r in rows):
            raise RagError('Incomplete recovery must finish before final scoring')
        return evaluate(cfg,store,stage['run_id'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    args=ap.parse_args(argv)
    from dotenv import load_dotenv
    from experiment_setup import check_plan

    load_dotenv(Path(read_json(args.plan)["project_root"]) / ".env")
    check_plan(args.plan, require_credentials=True)
    if (args.plan.parent / "pipeline.lock").exists():
        raise RagError("Experiment is already active")
    lock=args.plan.parent/'pipeline.lock'
    import os

    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        print(json.dumps(score(args.plan), indent=2))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
