"""Check local inputs and prepare a fresh frozen experiment without provider requests."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from experiment_review._context import ImmutableStore

from financial_rag.chunking import current_index
from financial_rag.common import RagError, digest, read_json, read_jsonl, sha256, write_json
from financial_rag.config import ARMS, load_config
from financial_rag.context_capacity import Capacity
from financial_rag.dataset import dataset
from financial_rag.storage import Store

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = PROJECT / '.local/contract-v1/cohort-296'
DEFAULT_SELECTION = PROJECT / '.local/contract-v1/pilot-selection.json'


def freeze_code(output: Path) -> dict:
    """Small allowlisted source bundle; never copy runtime data or credentials."""
    files = []
    for name in ('src', 'tools', 'tests', 'config', 'profiles', '.agents'):
        files.extend(p for p in (PROJECT / name).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and 'docling-studio' not in p.parts and 'archive' not in p.parts
                     and p.suffix in {'.py', '.ps1', '.md', '.json', '.toml'})
    files.extend(PROJECT.glob('*.md'))
    files.extend(PROJECT / name for name in ('config.toml', 'config.offline.toml', 'pyproject.toml', 'uv.lock')
                 if (PROJECT / name).is_file())
    files = sorted(set(files))
    with zipfile.ZipFile(output / 'code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT).as_posix())
    return {'code_archive': 'code.zip',
            'release_files': [{'path': str(path), 'bytes': path.stat().st_size} for path in files]}


def project_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(PROJECT.resolve()):
        raise RagError(f'Experiment inputs and outputs must be inside the project: {path}')
    return path


def require_material(root: Path, names: tuple[str, ...], purpose: str, next_step: str) -> None:
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise RagError(f'Missing {purpose}: ' + ', '.join(missing) + '. ' + next_step)


def freeze_model_routes(settings: str, cfg) -> str:
    """Write resolved URLs, never environment references, into a new plan."""
    for name, role in tomllib.loads(settings).get('models', {}).items():
        if 'base_url_env' not in role:
            continue
        pattern = rf'(?ms)(^\[models\.{re.escape(name)}\][^\n]*\n)(.*?)(?=^\[|\Z)'
        def replace(match):
            body, count = re.subn(r'(?m)^base_url_env\s*=.*$',
                                 lambda _: 'base_url = ' + json.dumps(getattr(cfg.models, name).base_url),
                                 match[2])
            if count != 1:
                raise RagError(f'Use a standalone base_url_env line in [models.{name}]')
            return match[1] + body
        settings, count = re.subn(pattern, replace, settings)
        if count != 1:
            raise RagError(f'Use a standalone [models.{name}] table for environment routing')
    return settings


def check_inputs(config_path: Path, cohort: Path, selection: Path, *, full: bool = False) -> tuple[dict, dict]:
    config_path, cohort, selection = map(project_path, (config_path, cohort, selection))
    require_material(config_path.parent, (config_path.name,), 'experiment configuration',
                     'Pass --config with a configured TOML file; see OPERATIONS.md.')
    cfg = load_config(config_path)
    if not cfg.query.source_work_dir or not cfg.query.capacity_evidence_path:
        raise RagError('Configure the frozen source workspace and capacity evidence')
    source_root = project_path(cfg.query.source_work_dir)
    require_material(source_root, ('experiment.sqlite',), 'source database',
                     'Prepare reviewed document facts and the flat/structured/enriched indices, '
                     'then set query.source_work_dir. Git does not contain the historical source database; see OPERATIONS.md.')
    require_material(cohort, ('manifest.json', 'queries.jsonl', 'labels.jsonl'), 'fixed cohort',
                     'Supply the matching cohort with --cohort; keep its original questions and labels. '
                     'The formal driver does not derive or invent historical selections; see OPERATIONS.md.')
    if not full:
        require_material(selection.parent, (selection.name,), 'pilot selection',
                         'Pass --selection with an explicit fixed question list, or use --scope full for the complete cohort.')
    project_path(cfg.query.capacity_evidence_path)
    capacity = Capacity(cfg)
    proof_reference = capacity.proof.get('probe_response_sha256')
    if proof_reference:
        proof = project_path(config_path.parent / capacity.proof['source_reference'])
        if not proof.is_file():
            raise RagError('Capacity response proof missing')
    cohort_manifest = read_json(cohort / 'manifest.json')
    for name, receipt in cohort_manifest['derived_files'].items():
        if not (cohort / name).is_file():
            raise RagError(f'Cohort file missing: {name}')
    queries = read_jsonl(cohort / 'queries.jsonl')
    labels = read_jsonl(cohort / 'labels.jsonl')
    query_ids = [row['query_id'] for row in queries]
    if (len(query_ids) != cfg.dataset.expected_english_queries
            or len(set(query_ids)) != len(query_ids)
            or query_ids != cohort_manifest['retained_query_ids']
            or len(labels) != len(query_ids) or {row['query_id'] for row in labels} != set(query_ids)):
        raise RagError('Fixed questions, labels or their order do not match the cohort')
    selected = query_ids if full else read_json(selection)['query_ids']
    if not selected or (not full and len(selected) > 49) or len(set(selected)) != len(selected) or set(selected) - set(query_ids):
        raise RagError('This preparation entrypoint accepts only a fixed pilot of at most 49 questions')
    for name in ('embedding', 'chat', 'judge'):
        if not getattr(cfg.models, name).configured():
            raise RagError(f'Configure models.{name} before preparing an experiment')
    with ImmutableStore(source_root) as source:
        if source.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RagError('Source database integrity check failed')
        manifest = dataset(source, verify_pdfs=True)
        if manifest['id'] != cohort_manifest['source_dataset_id']:
            raise RagError('Cohort belongs to a different source dataset')
        if (manifest['counts']['documents'] != cfg.dataset.expected_documents
                or manifest['counts']['pages'] != cfg.dataset.expected_pages):
            raise RagError('Source document or page count differs from the configuration')
        indices = {}
        model_id = digest(cfg.models.embedding.identity())
        for kind in ('flat', 'structured', 'enriched'):
            index = current_index(cfg, source, kind)
            count = source.db.execute('SELECT COUNT(*) FROM chunks WHERE index_id=?', (index['id'],)).fetchone()[0]
            vectors = source.db.execute(
                'SELECT COUNT(*),MIN(e.dimensions),MAX(e.dimensions) FROM chunks c '
                'JOIN chunk_vectors m ON m.chunk_id=c.id AND m.model_id=? '
                'JOIN embeddings e ON e.key=m.embedding_key WHERE c.index_id=?',
                (model_id, index['id']),
            ).fetchone()
            expected_dim = cfg.models.embedding.request_params.get('dimensions')
            if vectors[0] != count or vectors[1] != vectors[2] or (expected_dim and vectors[1] != expected_dim):
                raise RagError(f'Frozen {kind} index has missing or incompatible embeddings')
            if source.db.execute(f'SELECT COUNT(*) FROM {source.fts_name(index["id"])}').fetchone()[0] != count:
                raise RagError(f'Frozen {kind} sparse index is incomplete')
            indices[kind] = {'id': index['id'], 'chunks': count, 'dimensions': vectors[1]}
        portable_source = source.portable(manifest)
    summary = {
        'local_inputs_ready': True, 'project_root': str(PROJECT), 'source_workspace': str(source_root),
        'cohort_questions': len(queries), 'selected_questions': len(selected), 'scope': 'full' if full else 'pilot',
        'indices': indices,
        'context_window_tokens': capacity.limit, 'output_reserve_tokens': capacity.output,
        'credentials_present': {name: bool(os.environ.get(getattr(cfg.models, name).api_key_env))
                                if getattr(cfg.models, name).api_key_env else True
                                for name in ('embedding', 'chat', 'judge')},
        'provider_connectivity_checked': False, 'provider_requests': 0, 'full_run_authorized': False,
    }
    return summary, {'config': cfg, 'source_manifest': portable_source, 'query_ids': selected}


def prepare(config_path: Path, cohort: Path, selection: Path, output: Path, run_id: str,
            *, full: bool = False, full_run_authorized: bool = False,
            full_plan_authorized: bool = False, arms: list[str] | None = None,
            reuse_material: Path | None = None) -> Path:
    arms = list(ARMS) if arms is None else arms
    if not arms or len(arms)!=len(set(arms)) or set(arms)-set(ARMS):
        raise RagError('Select unique supported experimental arms')
    output = project_path(output)
    if output.parent != PROJECT / '.local/experiments' or output.exists():
        raise RagError('Choose a new direct child of .local/experiments; existing directories cannot be reused')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise RagError('Invalid run ID')
    if full and not (full_run_authorized or full_plan_authorized):
        raise RagError('Preparing an authorized full plan requires a new explicit user decision')
    if (full_run_authorized or full_plan_authorized) and not full:
        raise RagError('Full authorization applies only to a full plan')
    for previous in (PROJECT / '.local/experiments').glob('*/plan.json'):
        if any(stage['run_id'] == run_id for stage in read_json(previous).get('stages', [])):
            raise RagError('Choose a new run ID; existing IDs are reserved for their own experiment')
    summary, inputs = check_inputs(config_path, cohort, selection, full=full)
    cfg = inputs['config']
    reused = None
    if reuse_material is not None:
        if not full or arms != ['A','B','C']:
            raise RagError('Reused D/pre-D delivery requires a new ABC full plan')
        from experiment_reuse import verify
        reuse_material = project_path(reuse_material)
        require_material(reuse_material, ('manifest.json', 'reuse.sqlite', 'queries.jsonl', 'labels.jsonl'),
                         'D/pre-D reuse material',
                         'Provide the original extracted material with --reuse-material, or omit that option '
                         'for a new run without historical D/pre-D results. Never fabricate reused scores.')
        verify(reuse_material)
        reuse_manifest = read_json(reuse_material/'manifest.json')
        if (set(reuse_manifest['query_ids']) != set(inputs['query_ids'])):
            raise RagError('Reused D/pre-D cohort or source differs')
        for name in ('queries.jsonl','labels.jsonl'):
            if read_jsonl(reuse_material/name) != read_jsonl(cohort/name):
                raise RagError('Reused D/pre-D questions or labels differ')
        reused = {'path':str(reuse_material),'source_run_id':reuse_manifest['source_run_id']}
    settings = freeze_model_routes(config_path.read_text(encoding='utf-8'), cfg)
    output.mkdir(parents=True)
    prepared = output / 'runtime/prepared'
    prepared.mkdir(parents=True)
    for name in ('queries.jsonl', 'labels.jsonl', 'manifest.json'):
        shutil.copy2(cohort / name, prepared / name)
    # Resolve routes and paths before moving; model parameters and strategy stay intact.
    settings = re.sub(r'(?m)^work_dir\s*=.*$', 'work_dir = "runtime"', settings, count=1)
    for key, value in [('source_work_dir', cfg.query.source_work_dir),
                       ('capacity_evidence_path', cfg.query.capacity_evidence_path)]:
        settings = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = "{value.as_posix()}"', settings, count=1)
    for key, value in [('artifacts_path', cfg.parsing.artifacts_path),
                       ('ledger', cfg.validation.ledger), ('full_ledger', cfg.validation.full_ledger)]:
        if value:
            settings = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = "{Path(value).as_posix()}"', settings, count=1)
    config_copy = output / 'experiment.toml'
    config_copy.write_text(settings, encoding='utf-8')
    frozen_cfg = load_config(config_copy)
    if frozen_cfg.portable_settings() != cfg.portable_settings():
        raise RagError('Moving the configuration changed its behavioral settings')
    manifest = copy.deepcopy(inputs['source_manifest'])
    # Keep the runtime manifest's local PDF locators valid without duplicating bytes.
    # These are hard links to the preserved source, never links into an old experiment.
    for doc in manifest['documents']:
        source_pdf = project_path(cfg.query.source_work_dir / doc['pdf_path'])
        runtime_pdf = project_path(frozen_cfg.work_dir / doc['pdf_path'])
        if not runtime_pdf.is_relative_to(frozen_cfg.work_dir.resolve()):
            raise RagError('Runtime PDF locator escapes the new workspace')
        if not source_pdf.is_file():
            raise RagError('Source PDF missing during preparation')
        runtime_pdf.parent.mkdir(parents=True,exist_ok=True)
        os.link(source_pdf,runtime_pdf)
    manifest.update(
        id=digest({'source': manifest['id'], 'queries': sha256(prepared / 'queries.jsonl'),
                   'labels': sha256(prepared / 'labels.jsonl')}),
        document_only=False, prepared_dir=str(prepared), repo=cfg.dataset.repo, revision=cfg.dataset.revision,
        annotation_policy='Fixed cohort; labels are unchanged and consumed only by evaluation.',
    )
    manifest['counts']['queries'] = cfg.dataset.expected_english_queries
    with Store(frozen_cfg.work_dir) as runtime:
        runtime.set_meta('dataset', manifest)
        assert all(runtime.db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
                   for table in ('runs', 'records', 'cache', 'api_calls', 'embeddings'))
    plan = {
        'hash_verification': False, 'kind': 'maintained_frozen_experiment', 'scope': 'full' if full else 'pilot',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(PROJECT), 'config': str(config_copy),
        'source_workspace': str(cfg.query.source_work_dir),
        'cohort_files': [{'path': str(prepared / name)}
                         for name in ('queries.jsonl', 'labels.jsonl', 'manifest.json')],
        'selection_file': str((cohort / 'manifest.json' if full else selection).resolve()),
        'indices': summary['indices'], 'stages': [{'run_id': run_id, 'query_ids': inputs['query_ids'],'arms':arms}],
        'api_start': 0, 'runtime_seed': 'Dataset metadata and unchanged cohort only; no answers, judge cache or API history.',
        'full_plan_authorized': full and (full_plan_authorized or full_run_authorized),
        'full_run_authorized': full_run_authorized, 'await_user_full_decision': not full_run_authorized,
    }
    plan.update(freeze_code(output))
    if reused:
        plan['reuse_material'] = reused
    capacity = Capacity(cfg)
    if capacity.proof.get('probe_response_sha256'):
        plan['capacity_proof'] = {
            'path': str(project_path(config_path.parent / capacity.proof['source_reference'])),
        }
    for name in ('pilot-selection.json', 'development-overlap.json'):
        path = PROJECT / '.local/contract-v1' / name
        if path.is_file():
            shutil.copy2(path, output / name)
    plan['development_files'] = [{'path': str(output / name)}
                                 for name in ('pilot-selection.json', 'development-overlap.json')
                                 if (output / name).is_file()]
    path = output / 'plan.json'
    write_json(path, plan)
    write_json(output / 'status.json', {'phase': 'prepared_not_started', 'stages': {}, 'provider_requests': 0})
    return path


def check_plan(path: Path, *, require_credentials: bool = False) -> dict:
    path = project_path(path)
    plan = read_json(path)
    if plan.get('kind') != 'maintained_frozen_experiment':
        raise RagError('Historical plans are read-only. Prepare a fresh pilot with tools/experiment_setup.py.')
    cfg = load_config(Path(plan['config']))
    if plan.get('reuse_material'):
        from experiment_reuse import verify
        material=plan['reuse_material']
        require_material(Path(material['path']), ('manifest.json', 'reuse.sqlite', 'queries.jsonl', 'labels.jsonl'),
                         'D/pre-D reuse material', 'Restore the original material at the plan path before reviewing this delivery.')
        verify(Path(material['path']))
    if cfg.work_dir != path.parent / 'runtime':
        raise RagError('Runtime must be isolated inside the new experiment directory')
    Capacity(cfg)
    if len(plan['stages']) != 1:
        raise RagError('Use one independently frozen stage per experiment')
    is_full = plan.get('scope') == 'full'
    arms=plan['stages'][0].get('arms',list(ARMS))
    if not arms or len(arms)!=len(set(arms)) or set(arms)-set(ARMS):
        raise RagError('Invalid frozen arm selection')
    if (plan.get('scope') not in ('pilot', 'full')
            or not plan['stages'][0]['query_ids']
            or (not is_full and len(plan['stages'][0]['query_ids']) > 49)
            or (plan['full_run_authorized'] and not is_full)
            or (is_full and not (plan.get('full_plan_authorized') or plan['full_run_authorized']))
            or plan['await_user_full_decision'] == plan['full_run_authorized']):
        raise RagError('Invalid experiment scope or full authorization')
    # A resumable runtime may contain committed WAL after abrupt process exit.
    # Normal read-only SQLite includes that WAL; immutable archive access does not.
    # Never delete WAL or checkpoint it merely to make this check pass.
    with Store(cfg.work_dir, read_only=True) as runtime:
        if runtime.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RagError('Runtime database integrity check failed; preserve its DB/WAL/SHM')
        manifest = dataset(runtime, verify_queries=True, verify_pdfs=True)
        selected = plan['stages'][0]['query_ids']
        all_ids = [q['query_id'] for q in read_jsonl(Path(manifest['prepared_dir']) / 'queries.jsonl')]
        if len(set(selected)) != len(selected) or set(selected) - set(all_ids) or (is_full and selected != all_ids):
            raise RagError('Invalid fixed selection')
        if not is_full and selected != read_json(Path(plan['selection_file']))['query_ids']:
            raise RagError('Plan selection differs from the frozen development list')
    if require_credentials:
        for name in ('embedding', 'chat', 'judge'):
            getattr(cfg.models, name).require(name)
    return {'plan_ready': True, 'plan': str(path), 'phase': read_json(path.parent / 'status.json')['phase'],
            'selected_questions': len(selected), 'arms':arms, 'scope': plan['scope'], 'provider_requests': 0,
            'full_run_authorized': plan['full_run_authorized']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.toml')
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    parser.add_argument('--reuse-material',type=Path,help='Verified D/pre-D material for a new ABC full delivery')
    parser.add_argument('--full-run-authorized', action='store_true',
                        help='Record a separately obtained explicit user decision; never infer it from a readiness check')
    parser.add_argument('--full-plan-authorized', action='store_true',
                        help='Prepare a full plan only; execution remains disabled without separate authorization')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Read-only local readiness check; no provider requests')
    mode.add_argument('--output', type=Path, help='New directory directly inside .local/experiments')
    parser.add_argument('--run-id', help='New immutable pilot run ID; required with --output')
    args = parser.parse_args()
    try:
        if args.check:
            summary, inputs = check_inputs(args.config, args.cohort, args.selection, full=args.scope == 'full')
            if args.reuse_material:
                if args.scope != 'full' or args.arms != ['A', 'B', 'C']:
                    raise RagError('Reused D/pre-D delivery requires a new ABC full plan')
                material = project_path(args.reuse_material)
                require_material(material, ('manifest.json', 'reuse.sqlite', 'queries.jsonl', 'labels.jsonl'),
                                 'D/pre-D reuse material', 'Supply the original extracted package with --reuse-material; see OPERATIONS.md.')
                from experiment_reuse import verify
                summary['reuse_verification'] = verify(material)
                if set(read_json(material/'manifest.json')['query_ids']) != set(inputs['query_ids']):
                    raise RagError('Reused D/pre-D cohort differs from the selected questions')
                for name in ('queries.jsonl', 'labels.jsonl'):
                    if read_jsonl(material/name) != read_jsonl(args.cohort/name):
                        raise RagError('Reused D/pre-D questions or labels differ')
        else:
            if not args.run_id:
                parser.error('--output requires --run-id')
            path = prepare(args.config, args.cohort, args.selection, args.output, args.run_id,
                           full=args.scope == 'full', full_run_authorized=args.full_run_authorized,
                           full_plan_authorized=args.full_plan_authorized,arms=args.arms,reuse_material=args.reuse_material)
            summary = check_plan(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (RagError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
