"""One durable request ceiling shared by every work directory in a validation batch."""

import sqlite3
from datetime import datetime, timezone

from .common import RagError


class BudgetExhausted(RagError):
    """A durable guard has stopped the experiment before another HTTP call."""


def reserve(config, context, role):
    full = config.validation.full_ledger
    reservation = None
    if full is not None:
        batch = config.validation.full_batch
        limit = config.validation.full_request_limit
        if not batch or limit is None:
            raise RagError("Formal budget requires a batch and request limit.")
        full.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(full, timeout=30) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS full_batches(
                    id TEXT PRIMARY KEY, request_limit INTEGER NOT NULL,
                    token_limit INTEGER, unknown_limit INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS full_attempts(
                    batch TEXT NOT NULL, ordinal INTEGER NOT NULL, context TEXT NOT NULL,
                    role TEXT NOT NULL, reserved_at TEXT NOT NULL,
                    status_code INTEGER, total_tokens INTEGER, usage_known INTEGER,
                    PRIMARY KEY(batch,ordinal));
            """)
            db.execute("BEGIN IMMEDIATE")
            policy = (batch, limit, config.validation.full_token_limit,
                      config.validation.full_unknown_usage_limit)
            db.execute("INSERT OR IGNORE INTO full_batches VALUES (?,?,?,?)", policy)
            if db.execute("SELECT * FROM full_batches WHERE id=?", (batch,)).fetchone() != policy:
                raise BudgetExhausted("Formal budget policy changed for this batch.")
            used, tokens, unknown = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(total_tokens),0),"
                " COALESCE(SUM(CASE WHEN usage_known IS NOT 1 THEN 1 ELSE 0 END),0)"
                " FROM full_attempts WHERE batch=?", (batch,)
            ).fetchone()
            if used >= limit or (policy[2] is not None and tokens >= policy[2]) or unknown > policy[3]:
                raise BudgetExhausted(
                    f"Formal budget stopped before HTTP: requests={used}/{limit}, "
                    f"reported_tokens={tokens}/{policy[2]}, unknown_usage={unknown}/{policy[3]}."
                )
            reservation = (batch, used + 1)
            db.execute("INSERT INTO full_attempts(batch,ordinal,context,role,reserved_at)"
                       " VALUES (?,?,?,?,?)",
                       (*reservation, context, role, datetime.now(timezone.utc).isoformat()))
    path = config.validation.ledger
    if path is None or role == "embedding":
        return reservation
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,request_limit INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(
                batch TEXT NOT NULL, ordinal INTEGER NOT NULL, context TEXT NOT NULL,
                role TEXT NOT NULL, reserved_at TEXT NOT NULL, PRIMARY KEY(batch,ordinal));
        """)
        db.execute("BEGIN IMMEDIATE")
        batch, limit = config.validation.batch, config.validation.deepseek_request_limit
        db.execute("INSERT OR IGNORE INTO batches VALUES (?,?)", (batch, limit))
        if db.execute("SELECT request_limit FROM batches WHERE id=?", (batch,)).fetchone()[0] != limit:
            raise RagError("This validation batch already binds another request ceiling.")
        used = db.execute("SELECT COUNT(*) FROM attempts WHERE batch=?", (batch,)).fetchone()[0]
        if used >= limit:
            raise RagError(f"Validation request ceiling reached ({used}/{limit}); no HTTP request sent.")
        db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?)",
            (batch, used + 1, context, role, datetime.now(timezone.utc).isoformat()),
        )
    return reservation


def complete(config, reservation, status_code, usage):
    """Persist provider totals for the next reservation; unknown stays explicit."""
    if reservation is None:
        return
    total = usage.get("total_tokens") if usage else None
    if total is None and usage:
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            total = prompt + completion
        elif isinstance(prompt, int):
            total = prompt
    with sqlite3.connect(config.validation.full_ledger, timeout=30) as db:
        db.execute("UPDATE full_attempts SET status_code=?,total_tokens=?,usage_known=?"
                   " WHERE batch=? AND ordinal=?",
                   (status_code, total, int(total is not None), *reservation))


def status(config):
    path = config.validation.ledger
    used = 0
    if path and path.exists():
        with sqlite3.connect(path) as db:
            used = db.execute(
                "SELECT COUNT(*) FROM attempts WHERE batch=?", (config.validation.batch,)
            ).fetchone()[0]
    result = {
        "enabled": path is not None,
        "batch": config.validation.batch,
        "attempts_reserved": used,
        "limit": config.validation.deepseek_request_limit,
        "note": "Reservations precede HTTP, include retries, and survive crashes/resumes.",
    }
    full = config.validation.full_ledger
    if full is not None:
        totals = (0, 0, 0)
        if full.exists():
            with sqlite3.connect(full) as db:
                totals = db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(total_tokens),0),"
                    " COALESCE(SUM(CASE WHEN usage_known IS NOT 1 THEN 1 ELSE 0 END),0)"
                    " FROM full_attempts WHERE batch=?",
                    (config.validation.full_batch,),
                ).fetchone()
        result["formal"] = {
            "batch": config.validation.full_batch,
            "requests": totals[0], "request_limit": config.validation.full_request_limit,
            "reported_tokens": totals[1], "token_limit": config.validation.full_token_limit,
            "unknown_usage": totals[2],
            "unknown_usage_limit": config.validation.full_unknown_usage_limit,
        }
    return result
