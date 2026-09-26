from __future__ import annotations

import json
import re
import sqlite3
import zlib
from pathlib import Path

import numpy as np

from .common import RagError, canonical

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS parses(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes(
    id TEXT PRIMARY KEY, parse_id TEXT NOT NULL REFERENCES parses(id),
    ref TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(parse_id, ref));
CREATE TABLE IF NOT EXISTS indices(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
    id TEXT PRIMARY KEY, index_id TEXT NOT NULL REFERENCES indices(id), payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_by_index ON chunks(index_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED, index_id UNINDEXED, text);
CREATE TABLE IF NOT EXISTS embeddings(
    key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS api_calls(
    id INTEGER PRIMARY KEY AUTOINCREMENT, context TEXT NOT NULL, role TEXT NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS calls_by_context ON api_calls(context);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records(
    run_id TEXT NOT NULL REFERENCES runs(id), arm TEXT NOT NULL, query_id TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY(run_id, arm, query_id));
CREATE TABLE IF NOT EXISTS snapshots(
    sha256 TEXT PRIMARY KEY, encoding TEXT NOT NULL CHECK(encoding='zlib'), data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS resources(
    locator TEXT PRIMARY KEY, sha256 TEXT NOT NULL, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS doc_items(
    id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    doc_version TEXT NOT NULL REFERENCES documents(id),
    parse_id TEXT NOT NULL REFERENCES parses(id),
    parent_id TEXT, section_id TEXT, kind TEXT NOT NULL,
    reading_order INTEGER, sibling_order INTEGER NOT NULL, depth INTEGER NOT NULL CHECK(depth>=0),
    section_path TEXT NOT NULL, raw_payload TEXT NOT NULL,
    UNIQUE(parse_id,id), UNIQUE(parse_id,reading_order),
    FOREIGN KEY(parse_id,parent_id) REFERENCES doc_items(parse_id,id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(parse_id,section_id) REFERENCES doc_items(parse_id,id) DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX IF NOT EXISTS items_parent ON doc_items(parse_id,parent_id,sibling_order);
CREATE INDEX IF NOT EXISTS items_section ON doc_items(parse_id,section_id,reading_order);
CREATE TABLE IF NOT EXISTS item_relations(
    item_id TEXT NOT NULL REFERENCES doc_items(id) ON DELETE CASCADE,
    related_id TEXT NOT NULL REFERENCES doc_items(id) ON DELETE CASCADE,
    PRIMARY KEY(item_id,related_id));
CREATE TABLE IF NOT EXISTS chunk_items(
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL REFERENCES doc_items(id), ordinal INTEGER NOT NULL,
    PRIMARY KEY(chunk_id,item_id), UNIQUE(chunk_id,ordinal));
CREATE INDEX IF NOT EXISTS chunks_for_item ON chunk_items(item_id,chunk_id);
CREATE TABLE IF NOT EXISTS chunk_vectors(
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL, embedding_key TEXT NOT NULL REFERENCES embeddings(key),
    PRIMARY KEY(chunk_id,model_id));
CREATE INDEX IF NOT EXISTS vectors_for_model ON chunk_vectors(model_id,chunk_id);
CREATE TRIGGER IF NOT EXISTS item_document_binding BEFORE INSERT ON doc_items
WHEN NEW.doc_version != (SELECT json_extract(payload,'$.doc_version') FROM parses WHERE id=NEW.parse_id)
BEGIN SELECT RAISE(ABORT,'item document version differs from its parse'); END;
CREATE TRIGGER IF NOT EXISTS relation_version_binding BEFORE INSERT ON item_relations
WHEN (SELECT parse_id FROM doc_items WHERE id=NEW.item_id) !=
     (SELECT parse_id FROM doc_items WHERE id=NEW.related_id)
BEGIN SELECT RAISE(ABORT,'related items must share a fact version'); END;
CREATE TRIGGER IF NOT EXISTS chunk_version_binding BEFORE INSERT ON chunk_items
WHEN NOT EXISTS (
    SELECT 1 FROM chunks c JOIN indices i ON i.id=c.index_id, json_each(i.payload,'$.parse_ids') p
    WHERE c.id=NEW.chunk_id AND p.value=(SELECT parse_id FROM doc_items WHERE id=NEW.item_id))
BEGIN SELECT RAISE(ABORT,'chunk item outside its index fact versions'); END;
"""


class Store:
    def __init__(self, root: Path, read_only: bool = False):
        self.root = root.resolve()
        if read_only:
            self.db = sqlite3.connect((self.root / "experiment.sqlite").as_uri() + "?mode=ro", uri=True)
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(root / "experiment.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
        else:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        import sqlite_vec

        self.db.enable_load_extension(True)
        try:
            sqlite_vec.load(self.db)
        finally:
            self.db.enable_load_extension(False)
        if self.db.execute("SELECT vec_version()").fetchone()[0] != "v0.1.9":
            raise RagError("This schema requires sqlite-vec 0.1.9.")

    def resource(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise RagError("Resource locator escapes this work directory.")
        return path

    def portable(self, value):
        if isinstance(value, list):
            return [self.portable(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, val in value.items():
            if key in {"pdf_path", "snapshot", "prepared_dir"} and isinstance(val, str):
                result[key] = self.resource(val).relative_to(self.root).as_posix()
            else:
                result[key] = self.portable(val)
        return result

    def hydrate(self, value):
        if isinstance(value, list):
            return [self.hydrate(v) for v in value]
        if not isinstance(value, dict):
            return value
        return {
            key: str(self.resource(val))
            if key in {"pdf_path", "snapshot", "prepared_dir"} and isinstance(val, str)
            else self.hydrate(val)
            for key, val in value.items()
        }

    def save_snapshot(self, path: Path, checksum: str):
        from .common import sha256

        if sha256(path) != checksum:
            raise RagError("Snapshot checksum mismatch; refusing to store altered facts.")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?,?,?)",
                (checksum, "zlib", zlib.compress(path.read_bytes(), 6)),
            )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return self.hydrate(json.loads(row[0])) if row else default

    def set_meta(self, key: str, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, canonical(self.portable(value)))
            )

    def get(self, table: str, key: str):
        self._table(table)
        row = self.db.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()
        if row is None:
            raise RagError(f"Missing {table} artifact {key}; run its preceding stage.")
        return self.hydrate(json.loads(row[0]))

    @staticmethod
    def _table(table):
        if table not in {"documents", "parses", "indices", "chunks", "nodes", "runs"}:
            raise ValueError("Unknown artifact table")

    def put(self, table: str, key: str, value: dict):
        self._table(table)
        with self.db:
            self.db.execute(
                f"INSERT INTO {table}(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (key, canonical(self.portable(value))),
            )

    def put_parse(self, parsed: dict, nodes: list[dict]):
        from .facts import relations

        nodes = relations(nodes)
        payload = canonical(self.portable(parsed))
        existing = self.db.execute("SELECT payload FROM parses WHERE id=?", (parsed["id"],)).fetchone()
        if existing and existing[0] == payload:
            rows = self.db.execute(
                "SELECT id,payload FROM nodes WHERE parse_id=?", (parsed["id"],)
            ).fetchall()
            projected = self.db.execute(
                "SELECT COUNT(*) FROM doc_items WHERE parse_id=?", (parsed["id"],)
            ).fetchone()[0]
            expected = {n["id"]: canonical(n) for n in nodes}
            if projected == len(nodes) and {r["id"]: r["payload"] for r in rows} == expected:
                return
        with self.db:
            self.db.execute(
                "INSERT INTO parses VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (parsed["id"], canonical(self.portable(parsed))),
            )
            # A retry replaces a partial snapshot; nodes removed by the new
            # conversion must not survive from the earlier attempt.
            # A deliberate replacement invalidates dependent indices before removing facts.
            for row in self.db.execute("SELECT id,payload FROM indices").fetchall():
                if parsed["id"] in json.loads(row["payload"])["parse_ids"]:
                    self.drop_index(row["id"])
            self.db.execute("DELETE FROM nodes WHERE parse_id=?", (parsed["id"],))
            for node in nodes:
                self.db.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)",
                    (node["id"], parsed["id"], node["ref"], canonical(node)),
                )
                self.db.execute(
                    "INSERT INTO doc_items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        node["id"],
                        parsed["doc_version"],
                        parsed["id"],
                        node["parent_id"],
                        node["section_id"],
                        node["kind"],
                        node["reading_order"],
                        node["sibling_order"],
                        node["depth"],
                        canonical(node["headings"]),
                        canonical(node.get("raw", {})),
                    ),
                )
            for node in nodes:
                for ref in node.get("related_refs", []):
                    from .common import digest

                    self.db.execute(
                        "INSERT OR IGNORE INTO item_relations VALUES (?,?)",
                        (node["id"], digest([parsed["id"], ref])),
                    )

    def drop_index(self, iid: str):
        self.db.execute("DELETE FROM chunk_fts WHERE index_id=?", (iid,))
        self.db.execute("DELETE FROM chunks WHERE index_id=?", (iid,))
        self.db.execute("DROP TABLE IF EXISTS " + self.fts_name(iid))
        self.db.execute("DELETE FROM indices WHERE id=?", (iid,))
        self.db.execute("DELETE FROM metadata WHERE key LIKE 'index:%' AND value=?", (canonical(iid),))

    def nodes(self, parse_id: str) -> list[dict]:
        rows = self.db.execute("SELECT payload FROM nodes WHERE parse_id=? ORDER BY rowid", (parse_id,))
        return [json.loads(row[0]) for row in rows]

    def put_index(self, index: dict, chunks: list[dict]):
        # Commit the complete index and pointer together, never a half-built replacement.
        with self.db:
            self.db.execute(
                "INSERT INTO indices VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (index["id"], canonical(index)),
            )
            fts = self.fts_name(index["id"])
            self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5(id UNINDEXED,text)")
            self.db.execute(f"DELETE FROM {fts}")
            self.db.execute("DELETE FROM chunk_fts WHERE index_id=?", (index["id"],))
            self.db.execute("DELETE FROM chunks WHERE index_id=?", (index["id"],))
            for c in chunks:
                self.db.execute("INSERT INTO chunks VALUES (?,?,?)", (c["id"], index["id"], canonical(c)))
                self.db.execute("INSERT INTO chunk_fts VALUES (?,?,?)", (c["id"], index["id"], c["text"]))
                self.db.execute(f"INSERT INTO {fts} VALUES (?,?)", (c["id"], c["text"]))
                self.db.executemany(
                    "INSERT INTO chunk_items VALUES (?,?,?)",
                    [(c["id"], nid, i) for i, nid in enumerate(c["node_ids"])],
                )
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                ("index:" + index["kind"], canonical(index["id"])),
            )

    def chunks(self, index_id: str) -> list[dict]:
        return [
            json.loads(r[0])
            for r in self.db.execute("SELECT payload FROM chunks WHERE index_id=? ORDER BY id", (index_id,))
        ]

    @staticmethod
    def fts_name(index_id: str) -> str:
        if not re.fullmatch("[0-9a-f]{64}", index_id):
            raise ValueError("An FTS table name requires a SHA-256 index ID")
        return "fts_" + index_id

    def cached(self, key: str):
        row = self.db.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache(self, key: str, payload: dict):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, canonical(payload)))

    def vector(self, key: str) -> np.ndarray | None:
        row = self.db.execute("SELECT dimensions, vector FROM embeddings WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row[1], dtype="<f4").copy()
        if len(vec) != row[0] or not np.all(np.isfinite(vec)) or np.linalg.norm(vec) == 0:
            raise RagError("Corrupt cached embedding; inspect the SQLite embedding cache.")
        return vec

    def save_vectors(self, pairs: list[tuple[str, np.ndarray]]):
        with self.db:
            for key, vector in pairs:
                v = np.asarray(vector, dtype="<f4")
                if v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
                    raise RagError("Invalid embedding vector; batch was not cached.")
                self.db.execute(
                    "INSERT INTO embeddings VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET dimensions=excluded.dimensions,vector=excluded.vector",
                    (key, len(v), v.tobytes()),
                )

    def map_vectors(self, index_id: str, role):
        from .api import embedding_key
        from .common import digest

        model_id = digest(role.identity())
        with self.db:
            for c in self.chunks(index_id):
                key = embedding_key(role, c["text"])
                row = self.db.execute("SELECT dimensions FROM embeddings WHERE key=?", (key,)).fetchone()
                if not row:
                    raise RagError("Missing vectors; run embed for the current index and model.")
                expected = role.request_params.get("dimensions")
                if expected and row[0] != expected:
                    raise RagError("Cached embedding dimensions do not match the configured model.")
                self.db.execute(
                    "INSERT OR REPLACE INTO chunk_vectors VALUES (?,?,?)", (c["id"], model_id, key)
                )

    def dense(self, index_id: str, model_id: str, vector, limit: int):
        # Materialize the allowed index/model population BEFORE calculating distances.
        return self.db.execute(
            """
            WITH candidates AS MATERIALIZED (
                SELECT c.id,e.vector FROM chunks c
                JOIN chunk_vectors m ON m.chunk_id=c.id AND m.model_id=?
                JOIN embeddings e ON e.key=m.embedding_key
                WHERE c.index_id=? AND e.dimensions=?
            )
            SELECT id,vec_distance_cosine(vector,?) AS distance FROM candidates
            ORDER BY distance,id LIMIT ?
            """,
            (model_id, index_id, len(vector), np.asarray(vector, dtype="<f4").tobytes(), limit),
        ).fetchall()

    def log_call(self, context: str, role: str, payload: dict):
        with self.db:
            self.db.execute(
                "INSERT INTO api_calls(context,role,payload) VALUES (?,?,?)",
                (context, role, canonical(payload)),
            )

    def call_validation_error(self, context: str, role: str, error: str):
        self.annotate_call(context, role, {"error": error})

    def annotate_call(self, context: str, role: str, fields: dict):
        row = self.db.execute(
            "SELECT id,payload FROM api_calls WHERE context=? AND role=? ORDER BY id DESC LIMIT 1",
            (context, role),
        ).fetchone()
        if row:
            value = json.loads(row["payload"])
            value.update(fields)
            with self.db:
                self.db.execute("UPDATE api_calls SET payload=? WHERE id=?", (canonical(value), row["id"]))

    def calls(self, context: str) -> list[dict]:
        return [
            {"role": r[0], **json.loads(r[1])}
            for r in self.db.execute(
                "SELECT role,payload FROM api_calls WHERE context=? ORDER BY id", (context,)
            )
        ]

    def save_record(self, run_id: str, arm: str, query_id: str, payload: dict):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?)", (run_id, arm, query_id, canonical(payload))
            )

    def records(self, run_id: str) -> list[dict]:
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT payload FROM records WHERE run_id=? ORDER BY query_id,arm", (run_id,)
            )
        ]
