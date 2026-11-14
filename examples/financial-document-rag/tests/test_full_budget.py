from concurrent.futures import ThreadPoolExecutor

import pytest

from financial_rag.budget import BudgetExhausted, complete, reserve, status
from financial_rag.config import Config


def test_budget_counts_all_roles_and_rejects_without_a_reservation(tmp_path):
    config = Config()
    config.budget.ledger = tmp_path / "budget.sqlite"
    config.budget.batch = "example"
    config.budget.request_limit = 6
    config.budget.token_limit = 50
    for role in ("vision", "embedding", "chat", "chat", "judge", "judge"):
        complete(config, reserve(config, "run-1", role), 200, {"total_tokens": 9})
    before = status(config)
    with pytest.raises(BudgetExhausted, match="before HTTP"):
        reserve(config, "another-workdir", "embedding")
    assert status(config) == before
    assert before["attempts_reserved"] == 6
    assert before["reported_tokens"] == 54


def test_budget_stops_after_unknown_usage_and_rejects_policy_changes(tmp_path):
    config = Config()
    config.budget.ledger = tmp_path / "budget.sqlite"
    config.budget.request_limit = 10
    config.budget.unknown_usage_limit = 0
    complete(config, reserve(config, "run", "embedding"), 503, None)
    with pytest.raises(BudgetExhausted, match="unknown_usage"):
        reserve(config, "run", "chat")
    config.budget.unknown_usage_limit = 1
    with pytest.raises(BudgetExhausted, match="policy changed"):
        reserve(config, "run", "chat")


def test_concurrent_requests_share_one_ceiling(tmp_path):
    config = Config()
    config.budget.ledger = tmp_path / "budget.sqlite"
    config.budget.request_limit = 3

    def attempt(_):
        try:
            return reserve(config, "parallel", "chat")
        except BudgetExhausted:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        reservations = list(pool.map(attempt, range(12)))
    assert len([r for r in reservations if r is not None]) == 3
    assert status(config)["attempts_reserved"] == 3


def test_disabled_budget_creates_no_ledger():
    config = Config()
    assert reserve(config, "run", "chat") is None
    assert not status(config)["enabled"]
