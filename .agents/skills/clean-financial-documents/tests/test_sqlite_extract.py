import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "document_tools.py"


def test_extract_generic_sqlite_without_mutating_source(tmp_path):
    database = tmp_path / "documents.sqlite"
    original = b'{"document":"example"}'
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE records (name TEXT, content BLOB)")
        connection.execute("INSERT INTO records VALUES (?, ?)", ("sample", zlib.compress(original)))
    output = tmp_path / "document.json"
    subprocess.run([sys.executable, str(TOOLS), "extract-sqlite", "--database", str(database),
                    "--query", "SELECT content FROM records WHERE name = ?", "--params", '["sample"]',
                    "--zlib", "--output", str(output)], check=True, cwd=tmp_path, capture_output=True)
    assert output.read_bytes() == original
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    result = subprocess.run([sys.executable, str(TOOLS), "extract-sqlite", "--database", str(database),
                             "--query", "DELETE FROM records RETURNING content", "--output", str(tmp_path / "bad")],
                            cwd=tmp_path, capture_output=True)
    assert result.returncode != 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
