import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from app.models import ScoreOptions


class QueueFull(Exception):
    pass


class JobStore:
    """Each method owns its connection; API, coordinator and child process can share SQLite."""

    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, input_name TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, progress INTEGER NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    options TEXT NOT NULL, error TEXT, artifacts TEXT NOT NULL DEFAULT '[]'
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)")

    @staticmethod
    def decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["options"] = json.loads(job["options"])
        job["artifacts"] = json.loads(job["artifacts"])
        return job

    def create(self, job_id: str, filename: str, input_name: str,
               options: ScoreOptions, capacity: int) -> dict:
        now = time.time()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE status = 'queued'").fetchone()[0]
            if count >= capacity:
                raise QueueFull
            db.execute("""
                INSERT INTO jobs (id, filename, input_name, status, stage, progress,
                                  created_at, updated_at, options)
                VALUES (?, ?, ?, 'queued', 'queued', 0, ?, ?, ?)
            """, (job_id, filename, input_name, now, now, options.model_dump_json()))
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with closing(self.connect()) as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def claim_next(self) -> dict | None:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status = 'processing', stage = 'preparing', progress = 5, updated_at = ? WHERE id = ?",
                       (time.time(), row["id"]))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())

    def update(self, job_id: str, **fields) -> None:
        allowed = {"status", "stage", "progress", "error", "artifacts"}
        if not fields.keys() <= allowed:
            raise ValueError("Unsupported job update")
        if "artifacts" in fields:
            fields["artifacts"] = json.dumps(fields["artifacts"])
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with closing(self.connect()) as db, db:
            db.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))

    def fail(self, job_id: str, message: str) -> None:
        self.update(job_id, status="failed", stage="failed", error=message, artifacts=[])

    def recover_interrupted(self) -> None:
        with closing(self.connect()) as db, db:
            db.execute("""
                UPDATE jobs SET status = 'failed', stage = 'failed', updated_at = ?,
                error = 'The server restarted during processing. Please upload the recording again.'
                WHERE status = 'processing'
            """, (time.time(),))

    def expired(self, cutoff: float) -> list[str]:
        with closing(self.connect()) as db:
            return [r[0] for r in db.execute(
                "SELECT id FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < ?",
                (cutoff,))]

    def delete(self, job_id: str) -> None:
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
