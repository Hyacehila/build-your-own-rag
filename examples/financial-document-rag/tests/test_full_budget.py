import pytest

from financial_rag.budget import complete, reserve, status
from financial_rag.common import RagError
from financial_rag.config import Config


def test_formal_ledger_covers_all_roles_and_retries(tmp_path):
    config = Config()
    guard = config.validation
    guard.full_ledger = tmp_path / "formal.sqlite"
    guard.full_batch = "run-1"
    guard.full_request_limit = 6
    guard.full_token_limit = 50
    guard.full_unknown_usage_limit = 1
    for role in ("vision", "embedding", "chat", "chat", "judge"):
        attempt = reserve(config, "run-1", role)
        complete(config, attempt, 200, {"total_tokens": 9})
    assert status(config)["formal"]["requests"] == 5
    assert status(config)["formal"]["reported_tokens"] == 45
    attempt = reserve(config, "run-1", "judge")
    complete(config, attempt, 200, {"total_tokens": 9})
    with pytest.raises(RagError, match="Formal budget stopped"):
        reserve(config, "run-1", "embedding")
    assert status(config)["formal"]["requests"] == 6


def test_formal_ledger_stops_after_unknown_provider_usage(tmp_path):
    config = Config()
    guard = config.validation
    guard.full_ledger = tmp_path / "formal.sqlite"
    guard.full_batch = "run-2"
    guard.full_request_limit = 10
    guard.full_unknown_usage_limit = 0
    attempt = reserve(config, "run-2", "embedding")
    complete(config, attempt, 503, None)
    with pytest.raises(RagError, match="unknown_usage"):
        reserve(config, "run-2", "embedding")
