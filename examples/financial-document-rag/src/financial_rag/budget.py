"""Optional durable budget shared by all model roles and retry attempts."""

import sqlite3
from datetime import datetime, timezone

from .common import RagError


class BudgetExhausted(RagError):
    """The budget stopped a request before it was sent."""


def _totals(db, batch):
    return db.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0), "
        "COALESCE(SUM(CASE WHEN usage_known IS NOT 1 THEN 1 ELSE 0 END), 0) "
        "FROM budget_attempts WHERE batch=?", (batch,)
    ).fetchone()


def reserve(config, context, role):
    budget = config.budget
    if budget.ledger is None:
        return None
    if budget.request_limit is None:
        raise RagError("Set budget.request_limit when using a budget ledger.")
    budget.ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(budget.ledger, timeout=30) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS budget_batches(
                id TEXT PRIMARY KEY, request_limit INTEGER NOT NULL,
                token_limit INTEGER, unknown_limit INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS budget_attempts(
                batch TEXT NOT NULL, ordinal INTEGER NOT NULL, context TEXT NOT NULL,
                role TEXT NOT NULL, reserved_at TEXT NOT NULL,
                status_code INTEGER, total_tokens INTEGER, usage_known INTEGER,
                PRIMARY KEY(batch, ordinal));
        """)
        db.execute("BEGIN IMMEDIATE")
        policy = (budget.batch, budget.request_limit, budget.token_limit, budget.unknown_usage_limit)
        db.execute("INSERT OR IGNORE INTO budget_batches VALUES (?, ?, ?, ?)", policy)
        if db.execute("SELECT * FROM budget_batches WHERE id=?", (budget.batch,)).fetchone() != policy:
            raise BudgetExhausted("Budget policy changed for this batch; choose a new batch.")
        used, tokens, unknown = _totals(db, budget.batch)
        if (used >= budget.request_limit
            or (budget.token_limit is not None and tokens >= budget.token_limit)
            or unknown > budget.unknown_usage_limit):
            raise BudgetExhausted(
                f"Budget ceiling reached before HTTP: requests={used}/{budget.request_limit}, "
                f"reported_tokens={tokens}/{budget.token_limit}, "
                f"unknown_usage={unknown}/{budget.unknown_usage_limit}."
            )
        reservation = (budget.batch, used + 1)
        db.execute(
            "INSERT INTO budget_attempts(batch, ordinal, context, role, reserved_at) VALUES (?, ?, ?, ?, ?)",
            (*reservation, context, role, datetime.now(timezone.utc).isoformat()),
        )
    return reservation


def complete(config, reservation, status_code, usage):
    if reservation is None:
        return
    total = usage.get("total_tokens") if usage else None
    if total is None and usage:
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            total = prompt + completion
        elif isinstance(prompt, int):
            total = prompt
    with sqlite3.connect(config.budget.ledger, timeout=30) as db:
        db.execute(
            "UPDATE budget_attempts SET status_code=?, total_tokens=?, usage_known=? WHERE batch=? AND ordinal=?",
            (status_code, total, int(total is not None), *reservation),
        )


def status(config):
    budget = config.budget
    totals = (0, 0, 0)
    if budget.ledger is not None and budget.ledger.exists():
        with sqlite3.connect(budget.ledger, timeout=30) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_attempts'").fetchone():
                totals = _totals(db, budget.batch)
    return {
        "enabled": budget.ledger is not None,
        "batch": budget.batch,
        "attempts_reserved": totals[0],
        "request_limit": budget.request_limit,
        "reported_tokens": totals[1],
        "token_limit": budget.token_limit,
        "unknown_usage": totals[2],
        "unknown_usage_limit": budget.unknown_usage_limit,
        "note": "Reservations include every role and retry; tokens are checked after provider reports.",
    }
