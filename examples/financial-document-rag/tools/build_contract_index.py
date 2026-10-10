"""Serial, resumable summary construction. No question/label files are opened."""
import argparse,os
from pathlib import Path
from dotenv import load_dotenv
from financial_rag.config import load_config
from financial_rag.common import write_json
from financial_rag.storage import Store
from financial_rag.enrichment import enrich
from financial_rag.chunking import chunk,current_index
from financial_rag.retrieval import embed_index
from financial_rag.summary_windows import body_windows,summary_key


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--project',type=Path,required=True)
    args=ap.parse_args();load_dotenv(args.project/'.env');cfg=load_config(args.config);root=args.config.parent
    lock=root/'build.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    def state(phase,**extra):write_json(root/'build-status.json',{'phase':phase,'pid':os.getpid(),'hash_verification':False,**extra})
    try:
        state('building_summaries')
        with Store(cfg.work_dir) as store:
            # Successful immutable summaries are cached; only missing windows retry.
            for attempt in range(4):
                result=enrich(cfg,store)
                write_json(root/f'build-attempt-{attempt}.json',result)
                if not result['failed']:break
            if result['failed']:raise RuntimeError('Summary coverage remains incomplete; do not publish a partial C index')
            state('publishing_index');index=chunk(cfg,store,'enriched')
            state('embedding_index',index_id=index['id']);vectors=embed_index(cfg,store,'enriched')
            base=current_index(cfg,store,'structured');windows=body_windows(store,base)
            assert all(store.cached(summary_key(cfg,w)) for w in windows)
            assert current_index(cfg,store,'enriched')['base_index_id']==base['id']
            store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            write_json(root/'index-release.json',{'hash_verification':False,
                       'B_index':base['id'],'C_index':index,'body_windows':len(windows),'vectors':vectors,
                       'summary_contract':'same-section-five-blocks-overlap-one-v1',
                       'source_database':str(cfg.work_dir/'experiment.sqlite')})
        state('index_ready')
    except Exception as e:
        state('stopped_for_review',error={'type':type(e).__name__,'message':str(e)});raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
