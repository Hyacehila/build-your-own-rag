"""Credential checks shared by current portable releases."""

from __future__ import annotations

import os
import zlib

from .common import RagError


def secret_scan(db, paths=()):
    """Inspect stored values and decompressed snapshots without reporting secrets."""
    secrets = [
        os.environ.get(name, "").encode()
        for name in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "MINERU_API_KEY")
    ]
    secrets = [secret for secret in secrets if secret]
    findings = []
    checked = 0

    def inspect(raw, location):
        nonlocal checked
        checked += 1
        if any(secret in raw for secret in secrets):
            findings.append(location)

    for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'fts_%' AND name NOT LIKE 'chunk_fts_%'"
    ):
        table = row[0]
        for values in db.execute('SELECT * FROM "' + table.replace('"', '""') + '"'):
            for value in values:
                if isinstance(value, str):
                    inspect(value.encode(), table)
                elif isinstance(value, bytes):
                    inspect(value, table)
                    if table == "snapshots":
                        inspect(zlib.decompress(value), "decompressed_snapshot")
    for path in paths:
        inspect(path.read_bytes(), path.name)
    if findings:
        raise RagError("Credential scan failed in: " + ", ".join(sorted(set(findings))))
    return {"checked_values": checked, "known_keys_checked": len(secrets), "matches": 0}
