"""Read archived experiments without changing their frozen files or SQLite sidecars."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from financial_rag.storage import Store

PROJECT = Path(__file__).resolve().parents[2]


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def open_database(path: Path) -> sqlite3.Connection:
    path = path.resolve(strict=True)
    wal = path.with_name(path.name + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f'Archive has an uncheckpointed WAL; stop before reviewing: {wal}')
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


class ImmutableStore(Store):
    """Use the existing resource reader with an immutable, query-only connection."""

    def __init__(self, root: Path, read_only: bool = True):
        if not read_only:
            raise ValueError('Experiment review cannot open a writable Store')
        self.root = root.resolve(strict=True)
        self.read_only = True
        self.db = open_database(self.root / 'experiment.sqlite')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        import sqlite_vec

        self.db.enable_load_extension(True)
        try:
            sqlite_vec.load(self.db)
        finally:
            self.db.enable_load_extension(False)
