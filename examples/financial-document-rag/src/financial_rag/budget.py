"""One durable request ceiling shared by every work directory in a validation batch."""

import sqlite3
from datetime import datetime, timezone

from .common import RagError


def reserve(config, context, role):
    path = config.validation.ledger
    if path is None or role == "embedding":
        return
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


def status(config):
    path = config.validation.ledger
    used = 0
    if path and path.exists():
        with sqlite3.connect(path) as db:
            used = db.execute(
                "SELECT COUNT(*) FROM attempts WHERE batch=?", (config.validation.batch,)
            ).fetchone()[0]
    return {
        "enabled": path is not None,
        "batch": config.validation.batch,
        "attempts_reserved": used,
        "limit": config.validation.deepseek_request_limit,
        "note": "Reservations precede HTTP, include retries, and survive crashes/resumes.",
    }
