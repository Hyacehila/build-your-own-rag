"""Configuration and missing-input checks without local data or provider requests."""
import importlib
import json
from pathlib import Path

import pytest

from financial_rag.common import RagError
from financial_rag.config import Config, Role, load_config


@pytest.fixture
def setup_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    setup = importlib.import_module('experiment_setup')
    monkeypatch.setattr(setup, 'PROJECT', tmp_path)
    return setup


def test_environment_route_resolves_and_freezes_without_identity_change(tmp_path, monkeypatch, setup_module):
    raw = '[models.chat]\nbase_url_env = "TEST_RAG_URL"\nmodel = "fixed-model"\n'
    path = tmp_path / 'config.toml'
    path.write_text(raw)
    (tmp_path / '.env').write_text('TEST_RAG_URL=https://dotenv.example/v1\n')
    monkeypatch.setenv('TEST_RAG_URL', 'https://first.example/v1')
    cfg = load_config(path)
    assert cfg.models.chat.identity() == Role(base_url='https://first.example/v1', model='fixed-model').identity()
    frozen = tmp_path / 'experiment.toml'
    frozen.write_text(setup_module.freeze_model_routes(raw, cfg))
    monkeypatch.setenv('TEST_RAG_URL', 'https://second.example/v1')
    assert load_config(frozen).portable_settings() == cfg.portable_settings()
    assert 'base_url_env' not in frozen.read_text()
    assert load_config(path).models.chat.base_url == 'https://second.example/v1'


def test_environment_route_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv('TEST_RAG_DOTENV_URL', raising=False)
    path = tmp_path / 'config.toml'
    path.write_text('[models.chat]\nbase_url_env="TEST_RAG_DOTENV_URL"\nmodel="fixed-model"')
    (tmp_path / '.env').write_text('TEST_RAG_DOTENV_URL=https://dotenv.example/v1\n')
    cfg = load_config(path)
    assert cfg.models.chat.base_url == 'https://dotenv.example/v1'
    # dotenv writes into os.environ; keep the test isolated.
    monkeypatch.delenv('TEST_RAG_DOTENV_URL')


@pytest.mark.parametrize('value', ['', 'ftp://example.org', 'https://user:private-value@example.org/v1',
                                 'https://example.org/v1?key=private-value'])
def test_missing_or_invalid_route_stops_without_echoing_secrets(tmp_path, monkeypatch, value):
    path = tmp_path / 'config.toml'
    path.write_text('[models.chat]\nbase_url_env="TEST_RAG_URL"\nmodel="fixed-model"')
    monkeypatch.setenv('TEST_RAG_URL', value)
    with pytest.raises(RagError, match='TEST_RAG_URL') as error:
        load_config(path)
    assert 'private-value' not in str(error.value)


def test_literal_routes_ignore_environment_and_offline_config_stays_offline(tmp_path, monkeypatch):
    monkeypatch.setenv('FINANCIAL_RAG_BASE_URL', 'https://new.example/v1')
    path = tmp_path / 'historical.toml'
    path.write_text('[models.chat]\nbase_url="https://original.example/v1"\nmodel="fixed-model"')
    assert load_config(path).models.chat.base_url == 'https://original.example/v1'
    path.write_text('work_dir="offline"')
    assert not load_config(path).models.chat.configured()


def test_conflicting_route_fields_fail(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_RAG_URL', 'https://new.example/v1')
    path = tmp_path / 'config.toml'
    path.write_text('[models.chat]\nbase_url="https://original.example/v1"\nbase_url_env="TEST_RAG_URL"')
    with pytest.raises(RagError, match='not both'):
        load_config(path)


def test_fresh_project_reports_source_then_cohort_without_creating_database(tmp_path, setup_module):
    path = tmp_path / 'config.toml'
    path.write_text('[query]\nsource_work_dir="source"\ncapacity_evidence_path="capacity.json"')
    for role in ('chat', 'embedding', 'judge'):
        with path.open('a') as stream:
            stream.write(f'\n[models.{role}]\nbase_url="https://example.org/v1"\nmodel="fixed-model"\n')
    with pytest.raises(RagError, match='Missing source database.*OPERATIONS'):
        setup_module.check_inputs(path, tmp_path/'cohort', tmp_path/'selection.json', full=True)
    assert not (tmp_path/'source').exists()
    (tmp_path/'source').mkdir()
    (tmp_path/'source/experiment.sqlite').touch()
    with pytest.raises(RagError, match='Missing fixed cohort.*--cohort'):
        setup_module.check_inputs(path, tmp_path/'cohort', tmp_path/'selection.json', full=True)


def test_missing_reuse_material_stops_before_creating_output(tmp_path, monkeypatch, setup_module):
    cfg = Config()
    monkeypatch.setattr(setup_module, 'check_inputs', lambda *a, **k: ({}, {'config': cfg}))
    with pytest.raises(RagError, match='Missing D/pre-D reuse material.*--reuse-material'):
        setup_module.prepare(tmp_path/'config.toml', tmp_path/'cohort', tmp_path/'selection.json',
                             tmp_path/'.local/experiments/fresh', 'fresh', full=True,
                             full_plan_authorized=True, arms=['A', 'B', 'C'], reuse_material=tmp_path/'reuse')
    assert not (tmp_path/'.local').exists()


def test_unrelated_gateway_cannot_use_historical_capacity(tmp_path):
    from financial_rag.context_capacity import Capacity
    proof = tmp_path / 'capacity.json'
    proof.write_text(json.dumps({'model': 'fixed-model', 'base_url': 'https://original.example/v1',
                                 'verification': 'provider_error_metadata', 'source_reference': 'old-probe'}))
    cfg = Config(models={'chat': {'model': 'fixed-model', 'base_url': 'https://new.example/v1'}},
                 query={'capacity_evidence_path': proof})
    with pytest.raises(RagError, match='does not attest'):
        Capacity(cfg)


def test_check_option_does_not_silently_ignore_missing_reuse(tmp_path, monkeypatch, setup_module, capsys):
    import sys
    monkeypatch.setattr(setup_module, 'check_inputs', lambda *a, **k: ({}, {'query_ids': ['q']}))
    monkeypatch.setattr(sys, 'argv', ['experiment_setup.py', '--check', '--scope', 'full', '--arms',
                                    'A', 'B', 'C', '--reuse-material', str(tmp_path/'absent')])
    with pytest.raises(SystemExit) as error:
        setup_module.main()
    assert error.value.code == 2
    assert 'Missing D/pre-D reuse material' in capsys.readouterr().err
    assert not (tmp_path/'absent').exists()
