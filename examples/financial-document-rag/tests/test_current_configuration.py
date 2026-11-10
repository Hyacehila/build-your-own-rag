"""The maintained entrypoint must not silently select a retired experiment."""
import json
from pathlib import Path

import pytest

from financial_rag.config import Query, load_config
from financial_rag.context_capacity import Capacity

ROOT = Path(__file__).resolve().parents[1]


def test_main_configuration_is_the_accepted_strategy(monkeypatch):
    # The checked-in capacity record is historical; no live gateway is contacted.
    proof = json.loads((ROOT / 'config/gateway-capacity.json').read_text())
    monkeypatch.setenv('FINANCIAL_RAG_BASE_URL', proof['base_url'])
    cfg = load_config(ROOT / 'config.toml')
    assert (cfg.retrieval.sparse_k, cfg.retrieval.dense_k, cfg.retrieval.rrf_k, cfg.retrieval.final_k) == (50, 50, 20, 20)
    assert cfg.query.bc_assembly == 'BC-source-priority-v3'
    assert cfg.retrieval.flat_final_k==10 and cfg.retrieval.summary_supplement_k==5
    assert cfg.query.bc_text_tokens == 25000
    assert cfg.query.text_tokens == 40000  # A and D/pre-D keep their accepted initial packet.
    assert cfg.query.page_images == cfg.query.agent_request_images == 8
    assert cfg.query.agent_total_images == 16 and cfg.query.exploration_calls == 24
    assert cfg.chunking.table_headers == 'multirow' and cfg.chunking.source_scope == 'fragment'
    assert cfg.dataset.expected_english_queries == 296
    cap = Capacity(cfg)
    assert cap.limit == 131072 and cap.output == 16384 and cap.multiplier == 1.2
    proof = json.loads(cfg.query.capacity_evidence_path.read_text())
    assert proof['verification'] == 'provider_error_metadata'
    assert (ROOT / proof['source_reference']).is_file()


@pytest.mark.parametrize('options', [
    {'bc_assembly': 'BC-reading-order-eight-v1'},
    {'direct_read_hits': 20}, {'asset_read_fragments': 3},
])
def test_retired_strategy_and_unused_switches_are_rejected(options):
    with pytest.raises(ValueError):
        Query(**options)


def test_offline_configuration_has_same_strategy_without_provider_routes():
    cfg = load_config(ROOT / 'config.offline.toml')
    assert cfg.dataset.expected_english_queries == 296
    assert cfg.retrieval.rrf_k == 20 and cfg.query.bc_text_tokens == 25000
    assert cfg.query.bc_assembly == 'BC-source-priority-v3'
    assert cfg.retrieval.flat_final_k==10 and cfg.retrieval.summary_supplement_k==5
    assert all(not getattr(cfg.models, role).configured() for role in ('chat', 'judge', 'vision', 'embedding'))
