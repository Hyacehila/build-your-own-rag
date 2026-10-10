"""Archive review must not mutate frozen inputs, sidecars or scores."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/experiment_review/_context.py'
SPEC = importlib.util.spec_from_file_location('experiment_review_context', MODULE_PATH)
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)


def test_archive_connection_rejects_writes_and_preserves_database(tmp_path):
    database = tmp_path / 'experiment.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE saved(value INTEGER)')
        db.execute('INSERT INTO saved VALUES(24)')
    before = database.read_bytes()
    with review.open_database(database) as db:
        assert db.execute('SELECT value FROM saved').fetchone() == (24,)
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            db.execute('UPDATE saved SET value=29')
    assert database.read_bytes() == before
    assert not database.with_name('experiment.sqlite-shm').exists()
    assert not database.with_name('experiment.sqlite-wal').exists()


def test_nonempty_wal_requires_stopping_before_immutable_review(tmp_path):
    database = tmp_path / 'experiment.sqlite'
    database.touch()
    database.with_name(database.name + '-wal').write_bytes(b'uncheckpointed')
    with pytest.raises(ValueError, match='uncheckpointed WAL'):
        review.open_database(database)
