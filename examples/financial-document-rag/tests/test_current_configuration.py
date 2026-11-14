"""The checked-in configurations describe the maintained example."""

from pathlib import Path

from financial_rag.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_configuration_uses_a_new_workspace(monkeypatch):
    monkeypatch.delenv("FINANCIAL_RAG_BASE_URL", raising=False)
    cfg = load_config(ROOT / "config.toml")
    assert cfg.work_dir == (ROOT / ".local/workspace").resolve()
    assert Path(cfg.parsing.artifacts_path) == (ROOT / "model_weights/docling").resolve()
    assert cfg.query.source_work_dir is None
    assert cfg.query.context_window_tokens == 131072
    assert cfg.dataset.expected_english_queries == 296
    assert (cfg.retrieval.sparse_k, cfg.retrieval.dense_k, cfg.retrieval.rrf_k) == (50, 50, 20)
    assert cfg.retrieval.flat_final_k == 10 and cfg.retrieval.summary_supplement_k == 5
    assert cfg.query.bc_text_tokens == 25000
    assert cfg.chunking.source_scope == "fragment"


def test_offline_configuration_needs_no_model_service():
    cfg = load_config(ROOT / "config.offline.toml")
    assert cfg.dataset.expected_english_queries == 296
    assert cfg.tokenizer.kind == "approx"
    assert all(
        not getattr(cfg.models, role).configured() for role in ("chat", "judge", "vision", "embedding")
    )
