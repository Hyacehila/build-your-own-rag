"""Extract immutable D/pre-D results and their exact request/score provenance, without APIs."""
import argparse
import hashlib
import json
import sqlite3
import tomllib
import zlib
from pathlib import Path

from financial_rag.common import digest, read_json, write_json
from financial_rag.config import load_config
from financial_rag.storage import Store
from financial_rag.api import embedding_key
from experiment_review.new_run import judge_key


def unpack(data):
    return json.loads(zlib.decompress(data))


def retain_vectors(db, old, cfg):
    """Keep vectors actually used for the retained queries, with cache provenance."""
    db.execute('CREATE TABLE IF NOT EXISTS query_vectors(key TEXT PRIMARY KEY, dimensions INTEGER, vector BLOB, origin TEXT, sha256 TEXT)')
    texts=set()
    for data, in db.execute('SELECT data FROM records'):
        row=unpack(data)
        for attempt in [*row.get('previous_attempts',[]),row]:
            texts.update(t['query'] for t in attempt.get('trace',[]) if t.get('tool')=='search' and 'query' in t)
    with Store(cfg.query.source_work_dir,read_only=True) as source:
        for text in texts:
            key=embedding_key(cfg.models.embedding,text)
            value=old.db.execute('SELECT dimensions,vector FROM embeddings WHERE key=?',(key,)).fetchone()
            origin='historical_runtime'
            if value is None:
                value=source.db.execute('SELECT dimensions,vector FROM embeddings WHERE key=?',(key,)).fetchone()
                origin='frozen_source_cache'
            assert value is not None, ('Missing query vector',key)
            db.execute('INSERT OR IGNORE INTO query_vectors VALUES (?,?,?,?,?)',(key,value[0],value[1],origin,hashlib.sha256(value[1]).hexdigest()))


def extract(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    if output.exists():
        raise ValueError('Reuse destination must be new')
    plan = read_json(plan_path)
    # Read historical settings without coercing its retired B/C policy into a current one.
    cfg = load_config(Path(plan['project_root'])/'config.toml')
    historical = tomllib.loads(Path(plan['config']).read_text(encoding='utf-8'))
    cfg.models = type(cfg.models).model_validate(historical['models'])
    cfg.work_dir = (Path(plan['config']).parent/historical['work_dir']).resolve()
    run_id = plan['stages'][0]['run_id']
    ids = set(plan['stages'][0]['query_ids'])
    output.mkdir(parents=True)
    db = sqlite3.connect(output/'reuse.sqlite')
    db.executescript('''
        CREATE TABLE records(arm TEXT, query_id TEXT, sha256 TEXT, data BLOB, PRIMARY KEY(arm,query_id));
        CREATE TABLE scores(arm TEXT, query_id TEXT, sha256 TEXT, data BLOB, task_key TEXT, PRIMARY KEY(arm,query_id));
        CREATE TABLE cache(key TEXT PRIMARY KEY, sha256 TEXT, data BLOB);
        CREATE TABLE api_calls(id INTEGER PRIMARY KEY, context TEXT, role TEXT, payload TEXT);
        CREATE TABLE request_payloads(sha256 TEXT PRIMARY KEY, data BLOB);
        CREATE TABLE provenance(key TEXT PRIMARY KEY, data BLOB);
    ''')
    def blob(table, identity, raw, extra=()):
        data=raw.encode('utf-8'); values=(*identity, hashlib.sha256(data).hexdigest(), zlib.compress(data), *extra)
        db.execute(f'INSERT INTO {table} VALUES ({",".join("?" for _ in values)})', values)
    with Store(cfg.work_dir, read_only=True) as old:
        assert old.db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        run=old.get('runs',run_id)
        labels={r['query_id']:r for r in map(json.loads,(Path(run['dataset']['prepared_dir'])/'labels.jsonl').read_text(encoding='utf-8').splitlines())}
        original={}
        for arm,qid,raw in old.db.execute('SELECT arm,query_id,payload FROM records WHERE run_id=? AND arm IN (?,?)',(run_id,'D','pre-D')):
            row=json.loads(raw); original[arm,qid]=row
            blob('records',(arm,qid),raw)
        assert set(original)=={(a,q) for a in ('D','pre-D') for q in ids}
        tasks=set(); snapshots=set()
        with (cfg.work_dir/'runs'/run_id/'scored.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['arm'] not in ('D','pre-D'):continue
                old_row=original[row['arm'],row['query_id']]
                assert all(row[k]==v for k,v in old_row.items())
                assert isinstance(row['judge']['correct'],bool)
                key=judge_key(cfg,row,labels[row['query_id']]) if row['status']=='ok' else None
                if key:tasks.add(key)
                blob('scores',(row['arm'],row['query_id']),line.rstrip('\r\n'),(key,))
        for qid in ids:
            key=digest(['initial-evidence-v1',run_id,qid]); snapshots.add(key)
            value=old.cached(key)
            assert value and isinstance(value.get('packet'),dict)
            assert all(original[a,qid]['query']==value['packet']['question'] for a in ('D','pre-D'))
        for key in sorted(tasks|snapshots):
            raw=old.db.execute('SELECT payload FROM cache WHERE key=?',(key,)).fetchone()
            assert raw, ('Missing original cache',key)
            blob('cache',(key,),raw[0])
        contexts={'judge:'+key for key in tasks}; bodies=set()
        for row in old.db.execute('SELECT id,context,role,payload FROM api_calls ORDER BY id'):
            if row[1] not in contexts and not any(row[1].startswith(f'run:{run_id}:{a}:') for a in ('D','pre-D')):continue
            db.execute('INSERT INTO api_calls VALUES (?,?,?,?)',tuple(row))
            bodies.add(json.loads(row[3])['request_payload_sha256'])
        for key in sorted(bodies):
            value=old.db.execute('SELECT data FROM request_payloads WHERE sha256=?',(key,)).fetchone()
            assert value and isinstance(unpack(value[0]),dict)
            db.execute('INSERT INTO request_payloads VALUES (?,?)',(key,value[0]))
        # Keep exact original identities as provenance, never rewrite their run_id/paths.
        for key,raw in [('plan',json.dumps(plan)),('run',old.db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()[0]),
                        ('config',Path(plan['config']).read_text(encoding='utf-8'))]:
            db.execute('INSERT INTO provenance VALUES (?,?)',(key,zlib.compress(raw.encode())))
        for name in ('queries.jsonl','labels.jsonl'):
            data=(Path(run['dataset']['prepared_dir'])/name).read_bytes()
            (output/name).write_bytes(data)
        original_audit=plan_path.parent/'review/audit.json'
        if original_audit.exists():
            (output/'original-audit.json').write_bytes(original_audit.read_bytes())
        retain_vectors(db,old,cfg)
    db.commit();db.close()
    receipt={'source_run_id':run_id,'arms':['pre-D','D'],'query_ids':sorted(ids),
             'source_plan':str(plan_path),'hash_verification':False,
             'unknown_usage':'Original user pause at API boundary 4417 had an in-flight request with unknown submission/usage; absence of orphan bodies does not establish zero cost.',
             'historical_code_environment_retained':False,'generation_or_judging_performed':False,
             'files':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output/'manifest.json',receipt)
    result=verify(output)
    write_json(output/'verification.json',result)
    return result


def verify(root):
    root=Path(root); manifest=read_json(root/'manifest.json')
    if not isinstance(manifest.get('files'),list) or not manifest.get('arms') or not manifest.get('query_ids'):
        from financial_rag.common import RagError
        raise RagError('Invalid reuse manifest structure')
    for item in manifest['files']:
        assert (root/item['name']).is_file(),item['name']
    db=sqlite3.connect((root/'reuse.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    expected={(a,q) for a in manifest['arms'] for q in manifest['query_ids']}
    scores={}; records={}
    for table,target in [('records',records),('scores',scores)]:
        for a,q,h,b in db.execute(f'SELECT arm,query_id,sha256,data FROM {table}'):
            target[a,q]=unpack(b)
        assert set(target)==expected
    for a,q in expected:
        r,s=records[a,q],scores[a,q]
        assert all(s[k]==v for k,v in r.items()) and isinstance(s['judge']['correct'],bool)
        assert r['run_id']==manifest['source_run_id']
        if r['status']!='ok':assert r['attempt']==4 and not s['judge']['correct']
        snapshot_key=digest(['initial-evidence-v1',manifest['source_run_id'],q])
        v=unpack(db.execute('SELECT data FROM cache WHERE key=?',(snapshot_key,)).fetchone()[0])
        assert v['packet']['question']==r['query']
    for h,b in db.execute('SELECT sha256,data FROM cache'):
        assert isinstance(unpack(b),dict)
    for h,b in db.execute('SELECT sha256,data FROM request_payloads'):
        assert isinstance(unpack(b),dict)
    for h,b in db.execute('SELECT sha256,vector FROM query_vectors'):
        assert len(b)>0
    for payload, in db.execute('SELECT payload FROM api_calls'):
        assert db.execute('SELECT 1 FROM request_payloads WHERE sha256=?',(json.loads(payload)['request_payload_sha256'],)).fetchone()
    for key, in db.execute('SELECT DISTINCT task_key FROM scores WHERE task_key IS NOT NULL'):
        assert db.execute('SELECT 1 FROM cache WHERE key=?',(key,)).fetchone()
        assert db.execute('SELECT 1 FROM api_calls WHERE context=?',('judge:'+key,)).fetchone()
    result={'verified':True,'hash_verification':False,'records':len(records),'scores':len(scores),'snapshots':len(manifest['query_ids']),
            'correct':{a:sum(s['judge']['correct'] for (arm,q),s in scores.items() if arm==a) for a in manifest['arms']},
            'physical_requests':db.execute('SELECT COUNT(*) FROM api_calls').fetchone()[0],
            'query_vectors':db.execute('SELECT COUNT(*) FROM query_vectors').fetchone()[0],
            'provider_requests_performed':0,'retained_bytes':sum(p.stat().st_size for p in root.iterdir() if p.is_file())}
    db.close();return result


class SnapshotMaterial:
    """Read-only adapter for independent pre-D after the old full workspace is removed."""
    def __init__(self, root):
        root=Path(root);verify(root)
        self.manifest=read_json(root/'manifest.json')
        self.db=sqlite3.connect((root/'reuse.sqlite').resolve().as_uri()+'?mode=ro',uri=True)

    def cached(self,key):
        row=self.db.execute('SELECT data FROM cache WHERE key=?',(key,)).fetchone()
        return unpack(row[0]) if row else None

    def close(self):
        self.db.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true')
    args=ap.parse_args();print(json.dumps(verify(args.output) if args.verify else extract(args.plan,args.output),indent=2))
