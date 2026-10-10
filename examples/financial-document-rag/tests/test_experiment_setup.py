"""Exercise frozen preparation and the real driver against a loopback-only provider."""
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from financial_rag.chunking import chunk
from financial_rag.common import RagError, read_json, read_jsonl, sha256, write_json
from financial_rag.enrichment import enrich
from financial_rag.retrieval import embed_index
from financial_rag.storage import Store


@pytest.fixture
def prepared_inputs(configured, monkeypatch):
    cfg, source, fake = configured
    assert not enrich(cfg, source)['failed']
    chunk(cfg, source, 'enriched')
    for kind in ('flat', 'structured', 'enriched'):
        embed_index(cfg, source, kind)
    source.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    project = cfg.work_dir
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    setup = importlib.import_module('experiment_setup')
    monkeypatch.setattr(setup, 'PROJECT', project)
    driver = importlib.import_module('contract_experiment')
    (project / 'tools').mkdir()
    shutil.copy2(Path(driver.__file__), project / 'tools/contract_experiment.py')
    manifest = source.meta('dataset')
    cohort = project / 'cohort'
    cohort.mkdir()
    for name in ('queries.jsonl', 'labels.jsonl'):
        shutil.copy2(Path(manifest['prepared_dir']) / name, cohort / name)
    queries = read_jsonl(cohort / 'queries.jsonl')
    ids = [row['query_id'] for row in queries]
    write_json(cohort / 'manifest.json', {
        'source_dataset_id': manifest['id'], 'retained_query_ids': ids,
        'derived_files': {name: {'sha256': sha256(cohort / name)} for name in ('queries.jsonl', 'labels.jsonl')},
    })
    selection = project / 'selection.json'
    write_json(selection, {'query_ids': ids})
    lines = [f'work_dir = {json.dumps(str(project))}',
             '[dataset]', f'expected_documents = {manifest["counts"]["documents"]}',
             f'expected_pages = {manifest["counts"]["pages"]}', f'expected_english_queries = {len(ids)}',
             '[tokenizer]', 'kind = "approx"', 'name = "offline-test"', 'estimated = true',
             '[chunking]', 'max_tokens = 180', 'overlap = 25', '[query]',
             f'source_work_dir = {json.dumps(project.as_posix())}',
             f'capacity_evidence_path = {json.dumps(cfg.query.capacity_evidence_path.as_posix())}']
    for name in ('embedding', 'chat', 'vision', 'judge'):
        role = getattr(cfg.models, name)
        lines += [f'[models.{name}]', f'base_url = {json.dumps(role.base_url)}',
                  f'model = {json.dumps(role.model)}', 'retries = 0']
        if name == 'chat':
            lines += ['[models.chat.request_params]', 'max_tokens = 1024']
    config_path = project / 'config.toml'
    config_path.write_text('\n'.join(lines), encoding='utf-8')
    return setup, driver, source, fake, config_path, cohort, selection


def new_plan(inputs, name='fresh', arms=None):
    setup, _, _, _, config, cohort, selection = inputs
    return setup.prepare(config, cohort, selection, setup.PROJECT / '.local/experiments' / name, name, arms=arms)


def test_prepared_plan_freezes_environment_route(prepared_inputs, monkeypatch):
    from financial_rag.config import load_config
    setup, _, _, fake, path, _, _ = prepared_inputs
    original = load_config(path).portable_settings()
    text = path.read_text(encoding='utf-8')
    text = text.replace('base_url = ' + json.dumps(fake.url), 'base_url_env = "TEST_PLAN_GATEWAY"')
    path.write_text(text, encoding='utf-8')
    monkeypatch.setenv('TEST_PLAN_GATEWAY', fake.url)
    request_count = len(fake.requests)
    plan_path = new_plan(prepared_inputs)
    plan = read_json(plan_path)
    frozen = Path(plan['config'])
    assert 'base_url_env' not in frozen.read_text(encoding='utf-8')
    monkeypatch.setenv('TEST_PLAN_GATEWAY', 'https://different.example/v1')
    assert load_config(frozen).portable_settings() == original
    assert setup.check_plan(plan_path)['plan_ready']
    assert len(fake.requests) == request_count


def test_preparation_preserves_inputs_and_seeds_no_answers_or_requests(prepared_inputs):
    setup, _, source, fake, config, cohort, selection = prepared_inputs
    before = (source.root / 'experiment.sqlite').read_bytes()
    request_count = len(fake.requests)
    plan_path = new_plan(prepared_inputs)
    assert setup.check_plan(plan_path)['plan_ready']
    assert len(fake.requests) == request_count
    assert (source.root / 'experiment.sqlite').read_bytes() == before
    plan = read_json(plan_path)
    assert not plan['full_run_authorized'] and plan['await_user_full_decision']
    for name in ('queries.jsonl', 'labels.jsonl'):
        assert (plan_path.parent / 'runtime/prepared' / name).read_bytes() == (cohort / name).read_bytes()
    with Store(plan_path.parent / 'runtime', read_only=True) as runtime:
        from financial_rag.dataset import dataset
        assert dataset(runtime,verify_pdfs=True)['counts']['documents']>0
        for table in ('runs', 'records', 'cache', 'api_calls', 'embeddings'):
            assert runtime.db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
    with pytest.raises(RagError, match='existing directories'):
        new_plan(prepared_inputs)
    with pytest.raises(RagError, match='new run ID'):
        setup.prepare(config, cohort, selection, setup.PROJECT / '.local/experiments/another', 'fresh')


def test_config_comment_change_does_not_require_hash_verification(prepared_inputs):
    setup, driver, _, fake, _, _, _ = prepared_inputs
    plan_path = new_plan(prepared_inputs)
    original = read_json(plan_path)['config']
    config = Path(original)
    config.write_text(config.read_text() + '\n# changed after freezing\n')
    requests = len(fake.requests)
    assert setup.check_plan(plan_path)['plan_ready']
    assert len(fake.requests) == requests


def test_legacy_checksum_fields_are_not_execution_gates(prepared_inputs):
    setup, _, _, fake, _, _, _ = prepared_inputs
    path = new_plan(prepared_inputs)
    plan = read_json(path)
    assert plan['hash_verification'] is False
    assert 'frozen_code_files' not in plan and 'code_archive_sha256' not in plan
    plan.update(implementation='retired', config_sha256='retired',
                frozen_code_files=[{'path': 'missing-retired-source.py', 'sha256': 'retired'}])
    write_json(path, plan)
    before = len(fake.requests)
    assert setup.check_plan(path)['plan_ready']
    assert len(fake.requests) == before


def test_full_preparation_requires_separate_authorization(prepared_inputs):
    setup, _, _, fake, config, cohort, selection = prepared_inputs
    requests = len(fake.requests)
    with pytest.raises(RagError, match='explicit user decision'):
        setup.prepare(config, cohort, selection, setup.PROJECT / '.local/experiments/full', 'full', full=True)
    assert len(fake.requests) == requests
    assert not (setup.PROJECT / '.local/experiments/full').exists()


def test_changed_plan_selection_is_rejected(prepared_inputs):
    setup, _, _, _, _, _, _ = prepared_inputs
    path = new_plan(prepared_inputs)
    plan = read_json(path)
    plan['stages'][0]['query_ids'].pop()
    write_json(path, plan)
    with pytest.raises(RagError, match='frozen development list'):
        setup.check_plan(path)


def test_explicit_full_preparation_uses_the_entire_fixed_cohort(prepared_inputs):
    setup, _, _, fake, config, cohort, selection = prepared_inputs
    requests = len(fake.requests)
    path = setup.prepare(config, cohort, selection, setup.PROJECT / '.local/experiments/full', 'full',
                         full=True, full_run_authorized=True)
    checked = setup.check_plan(path)
    assert checked['scope'] == 'full' and checked['full_run_authorized']
    assert read_json(path)['stages'][0]['query_ids'] == read_json(cohort / 'manifest.json')['retained_query_ids']
    assert len(fake.requests) == requests


@pytest.mark.parametrize('arms',[None,['A','B','C']])
def test_real_driver_checks_then_runs_five_arms_and_final_scores(prepared_inputs, monkeypatch, capsys,arms):
    setup, driver, source, fake, _, _, _ = prepared_inputs
    plan_path = new_plan(prepared_inputs,arms=arms)
    requests = len(fake.requests)
    monkeypatch.setattr('sys.argv', ['contract_experiment.py', '--plan', str(plan_path), '--check'])
    driver.main()
    assert '"plan_ready": true' in capsys.readouterr().out
    assert len(fake.requests) == requests
    assert not (plan_path.parent / 'pipeline.lock').exists()
    monkeypatch.setattr('sys.argv', ['contract_experiment.py', '--plan', str(plan_path)])
    driver.main()
    state = read_json(plan_path.parent / 'status.json')
    assert state['phase'] == 'pilot_complete_awaiting_user_decision'
    assert state['stages']['fresh']['complete']
    with Store(plan_path.parent / 'runtime', read_only=True) as runtime:
        rows = runtime.records('fresh')
        assert len(rows) == (9 if arms else 15) and all(row['status'] == 'ok' for row in rows)
        scores = read_jsonl(plan_path.parent / 'runtime/runs/fresh/scored.jsonl')
        assert len(scores) == (9 if arms else 15) and all(isinstance(row['judge']['correct'], bool) for row in scores)
        if arms:assert {r['arm'] for r in rows}==set(arms)
    assert source.db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 0
    assert not (plan_path.parent / 'pipeline.lock').exists()


def test_committed_wal_survives_abrupt_exit_and_preflight(prepared_inputs):
    setup, _, _, fake, _, _, _ = prepared_inputs
    path = new_plan(prepared_inputs)
    database = path.parent / 'runtime/experiment.sqlite'
    script = """import os, sqlite3, sys
db=sqlite3.connect(sys.argv[1])
db.execute('PRAGMA journal_mode=WAL')
db.execute('PRAGMA wal_autocheckpoint=0')
db.execute('CREATE TABLE recovery_marker(value TEXT)')
db.execute("INSERT INTO recovery_marker VALUES('committed-before-crash')")
db.commit()
os._exit(17)
"""
    child = subprocess.run([sys.executable, '-B', '-c', script, str(database)], check=False)
    assert child.returncode == 17
    wal = database.with_name(database.name+'-wal')
    before = database.read_bytes(), wal.read_bytes()
    requests = len(fake.requests)
    assert setup.check_plan(path)['plan_ready']
    assert (database.read_bytes(), wal.read_bytes()) == before
    with Store(database.parent, read_only=True) as runtime:
        assert runtime.db.execute('SELECT value FROM recovery_marker').fetchone()[0] == 'committed-before-crash'
    assert len(fake.requests) == requests


def test_full_plan_only_can_be_checked_but_cannot_execute(prepared_inputs, monkeypatch):
    setup, driver, _, fake, config, cohort, selection = prepared_inputs
    path = setup.prepare(config, cohort, selection, setup.PROJECT/'.local/experiments/full-ready',
                         'full-ready', full=True, full_plan_authorized=True)
    assert setup.check_plan(path)['plan_ready']
    assert not read_json(path)['full_run_authorized']
    requests = len(fake.requests)
    monkeypatch.setattr('sys.argv', ['driver', '--plan', str(path)])
    with pytest.raises(RuntimeError, match='explicit user decision'):
        driver.main()
    assert len(fake.requests) == requests


def test_exhausted_failure_and_successful_retries_have_one_terminal_report(prepared_inputs, monkeypatch):
    _, driver, _, fake, _, _, _ = prepared_inputs
    from financial_rag.runner import QuestionRunner
    original = QuestionRunner.run
    counts = {}
    def flaky(self, question, arm, **kwargs):
        key = (arm, question['query_id'])
        counts[key] = counts.get(key, 0)+1
        result = original(self, question, arm, **kwargs)
        failures = {'fixture-revenue': 1, 'fixture-growth': 4, 'fixture-cash': 3}
        if arm == 'A' and counts[key] <= failures[question['query_id']]:
            result.update(status='failed', answer=None, error={'type':'SyntheticFailure','message':'offline fixture'})
        return result
    monkeypatch.setattr(QuestionRunner, 'run', flaky)
    path = new_plan(prepared_inputs)
    monkeypatch.setattr('sys.argv', ['driver','--plan',str(path)])
    driver.main()
    review = read_json(path.parent/'review/analysis.json')
    assert review['records'] == 15 and review['terminal_valid_answers'] == 14
    assert review['checks']['exhausted_terminal_records'] == 1
    assert review['checks']['successful_checkpoint_records_present'] > 0
    assert counts['A','fixture-growth'] == 4
    assert counts['A','fixture-cash'] == 4
    assert counts['A','fixture-revenue'] == 2
    assert all(n == 1 for (arm,q),n in counts.items() if arm != 'A')
    assert review['stats']['A']['n'] == 3
    requests = len(fake.requests)
    driver.main()
    assert len(fake.requests) == requests


def test_interrupted_generation_then_judge_failure_resumes_without_regeneration(prepared_inputs, monkeypatch):
    _, driver, _, fake, _, _, _ = prepared_inputs
    from financial_rag.runner import QuestionRunner
    from financial_rag.common import digest
    original = QuestionRunner.run
    counter = 0
    def interrupted(self, *args, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 2:
            raise KeyboardInterrupt('synthetic abrupt interruption between records')
        return original(self, *args, **kwargs)
    path = new_plan(prepared_inputs)
    monkeypatch.setattr('sys.argv', ['driver','--plan',str(path)])
    monkeypatch.setattr(QuestionRunner,'run',interrupted)
    with pytest.raises(KeyboardInterrupt):driver.main()
    with Store(path.parent/'runtime',read_only=True) as runtime:
        first = runtime.records('fresh')
    assert len(first) == 1 and first[0]['status'] == 'ok'
    monkeypatch.setattr(QuestionRunner,'run',original)
    import financial_rag.evaluation as evaluation
    api_chat = evaluation.ModelAPI.validated_chat
    def unavailable(self, role, *args, **kwargs):
        if role == 'judge':raise RuntimeError('Synthetic judge outage')
        return api_chat(self,role,*args,**kwargs)
    monkeypatch.setattr(evaluation.ModelAPI,'validated_chat',unavailable)
    with pytest.raises(RuntimeError,match='Evaluation service failed'):driver.main()
    with Store(path.parent/'runtime',read_only=True) as runtime:
        before = digest(runtime.records('fresh'))
        kept = next(r for r in runtime.records('fresh') if (r['arm'],r['query_id']) == (first[0]['arm'],first[0]['query_id']))
        assert digest(kept) == digest(first[0])
    monkeypatch.setattr(evaluation.ModelAPI,'validated_chat',api_chat)
    def forbid(*args,**kwargs):raise AssertionError('Successful answer regenerated')
    monkeypatch.setattr(QuestionRunner,'run',forbid)
    driver.main()
    with Store(path.parent/'runtime',read_only=True) as runtime:
        assert digest(runtime.records('fresh')) == before
    assert read_json(path.parent/'status.json')['stages']['fresh']['complete']
    from experiment_review.new_run import build_review
    request_count = len(fake.requests)
    assert build_review(path)['records'] == 15
    assert len(fake.requests) == request_count
    scored = path.parent/'runtime/runs/fresh/scored.jsonl'
    scored.write_text(scored.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    assert build_review(path)['records']==15  # Whitespace is not an integrity gate.
    data=read_jsonl(scored);data[0]['answer']='different answer'
    scored.write_text('\n'.join(json.dumps(r) for r in data),encoding='utf-8')
    with pytest.raises(RagError,match='Scored answer mismatch'):build_review(path)


def test_historical_plan_cannot_be_used_as_new_plan(prepared_inputs):
    setup, _, _, _, _, _, _ = prepared_inputs
    path = setup.PROJECT / 'archive-plan.json'
    write_json(path, {'project_root': str(setup.PROJECT), 'stages': []})
    with pytest.raises(RagError, match='Historical plans are read-only'):
        setup.check_plan(path)


def test_independent_schemes_share_run_identity_and_pred_requires_complete_snapshots(prepared_inputs, monkeypatch):
    from financial_rag.common import digest
    _,driver,_,fake,_,_,_=prepared_inputs
    path=new_plan(prepared_inputs)
    monkeypatch.setattr('sys.argv',['scheme','--plan',str(path)])
    before=len(fake.requests)
    with pytest.raises(RuntimeError,match='snapshot-plan'):driver.main(arm='pre-D')
    assert len(fake.requests)==before
    monkeypatch.setattr('sys.argv',['scheme','--plan',str(path),'--snapshot-plan',str(path)])
    with pytest.raises(RuntimeError,match='Complete matching'):driver.main(arm='pre-D')
    assert len(fake.requests)==before
    for arm in ('A','B','C','D'):
        monkeypatch.setattr('sys.argv',['scheme','--plan',str(path)])
        driver.main(arm=arm)
    with Store(path.parent/'runtime',read_only=True) as store:
        original={(r['arm'],r['query_id']):digest(r) for r in store.records('fresh')}
        assert len(original)==12
        assert not any(r['role']=='judge' for r in store.db.execute('SELECT role FROM api_calls'))
    monkeypatch.setattr('sys.argv',['scheme','--plan',str(path),'--snapshot-plan',str(path)])
    driver.main(arm='pre-D')
    before=len(fake.requests)
    driver.main(arm='pre-D')
    assert len(fake.requests)==before
    monkeypatch.setattr('sys.argv',['serial','--plan',str(path)])
    driver.main()
    with Store(path.parent/'runtime',read_only=True) as store:
        assert all(digest(r)==original[r['arm'],r['query_id']] for r in store.records('fresh') if r['arm']!='pre-D')
    assert read_json(path.parent/'review/analysis.json')['records']==15


def test_full_abc_reuse_delivery_preserves_original_records_and_never_rejudges(prepared_inputs,monkeypatch):
    from experiment_reuse import extract,verify
    from experiment_delivery import build_delivery
    from financial_rag.common import digest
    setup,driver,_,fake,config,cohort,selection=prepared_inputs
    old=new_plan(prepared_inputs,'original')
    monkeypatch.setattr('sys.argv',['driver','--plan',str(old)])
    driver.main()
    reused=setup.PROJECT/'retained'
    before=len(fake.requests)
    proof=extract(old,reused)
    assert proof['records']==6 and proof['snapshots']==3 and verify(reused)['verified']
    assert len(fake.requests)==before
    path=setup.prepare(config,cohort,selection,setup.PROJECT/'.local/experiments/abc','abc',full=True,full_run_authorized=True,arms=['A','B','C'],reuse_material=reused)
    monkeypatch.setattr('sys.argv',['driver','--plan',str(path)])
    with pytest.raises(RuntimeError,match='outside the frozen'):driver.main(arm='D')
    driver.main()
    report=read_json(path.parent/'delivery/analysis.json')
    assert report['new_records']==9 and report['reused_records']==6
    with Store(path.parent/'runtime',read_only=True) as store:
        assert {r['arm'] for r in store.records('abc')}=={'A','B','C'}
        assert not store.db.execute("SELECT 1 FROM api_calls WHERE context LIKE 'run:abc:D:%' OR context LIKE 'run:abc:pre-D:%'").fetchone()
        saved=digest(store.records('abc'))
    before=len(fake.requests)
    build_delivery(path);driver.main()
    assert len(fake.requests)==before and verify(reused)['verified']
    with Store(path.parent/'runtime',read_only=True) as store:assert digest(store.records('abc'))==saved
    standalone=new_plan(prepared_inputs,'pred-independent',arms=['pre-D'])
    monkeypatch.setattr('sys.argv',['pre-D','--plan',str(standalone),'--snapshot-material',str(reused)])
    driver.main(arm='pre-D')
    with Store(standalone.parent/'runtime',read_only=True) as store:
        assert len(store.records('pred-independent'))==3
        assert {r['arm'] for r in store.records('pred-independent')}=={'pre-D'}
        assert not store.db.execute("SELECT 1 FROM api_calls WHERE role='judge'").fetchone()
    assert verify(reused)['verified']


def test_invalid_reuse_structure_prevents_execution(prepared_inputs,monkeypatch):
    setup,_,_,fake,_,_,_=prepared_inputs
    path=new_plan(prepared_inputs,arms=['A','B','C'])
    p=read_json(path);p['reuse_material']={'path':str(path.parent),'manifest_sha256':'not-the-file-hash'}
    (path.parent/'manifest.json').write_text('{}');write_json(path,p)
    before=len(fake.requests)
    with pytest.raises(RagError,match='reuse manifest'):setup.check_plan(path)
    assert len(fake.requests)==before
