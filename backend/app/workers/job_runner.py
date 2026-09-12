import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

from app.config import Settings
from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


class JobRunner:
    """One durable queue, one isolated pipeline process at a time. Run one API worker."""

    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="job-coordinator", daemon=True)

    def start(self) -> None:
        self.store.recover_interrupted()
        # Recovery may follow a hard process kill, so also discard abandoned inputs.
        for directory in self.settings.jobs_dir.iterdir():
            if directory.is_dir():
                job = self.store.get(directory.name)
                if job and job["status"] in ("completed", "failed"):
                    self.remove_audio(job)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=10)

    def remove_audio(self, job: dict) -> None:
        directory = self.settings.jobs_dir / job["id"]
        (directory / job["input_name"]).unlink(missing_ok=True)
        shutil.rmtree(directory / 'work', ignore_errors=True)

    def cleanup(self) -> None:
        cutoff = time.time() - self.settings.retention_hours * 3600
        for job_id in self.store.expired(cutoff):
            shutil.rmtree(self.settings.jobs_dir / job_id, ignore_errors=True)
            self.store.delete(job_id)
        # Delete incomplete uploads left by a server crash before the DB insert.
        for directory in self.settings.jobs_dir.iterdir():
            # Legacy in-memory jobs used 32-character IDs; leave their stored files alone.
            try:
                if str(UUID(directory.name)) != directory.name:
                    continue
            except ValueError:
                continue
            if directory.is_dir() and directory.stat().st_mtime < cutoff and not self.store.get(directory.name):
                shutil.rmtree(directory, ignore_errors=True)

    def run(self) -> None:
        last_cleanup = 0.0
        while not self.stopping.is_set():
            try:
                if time.monotonic() - last_cleanup > 60:
                    self.cleanup()
                    last_cleanup = time.monotonic()
                if job := self.store.claim_next():
                    self.execute(job)
                else:
                    self.stopping.wait(0.5)
            except Exception:
                logger.exception("Job coordinator failed; retrying")
                self.stopping.wait(1)

    @staticmethod
    def kill(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

    def execute(self, job: dict) -> None:
        directory = self.settings.jobs_dir / job["id"]
        # Explicitly propagate effective settings (including test and programmatic config).
        environment = {**os.environ, **{
            key.upper(): (str(value) if isinstance(value, (str, int, float, Path))
                          else json.dumps(value))
            for key, value in self.settings.model_dump().items() if value is not None
        }, "DATA_DIR": str(self.settings.data_dir.resolve()), "TF_CPP_MIN_LOG_LEVEL": "2",
            "OMP_NUM_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2"}
        process = None
        try:
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen([sys.executable, "-m", "app.workers.transcription_worker", job["id"]],
                                           cwd=Path(__file__).resolve().parents[2],
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                deadline = (time.monotonic() + self.settings.job_timeout_seconds
                            if self.settings.job_timeout_seconds else None)
                while process.poll() is None:
                    stopped = self.stopping.wait(0.25)
                    if stopped or (deadline is not None and time.monotonic() > deadline):
                        self.kill(process)
                        self.store.fail(job["id"],
                                        "Processing was interrupted by a server shutdown. Please upload again."
                                        if stopped else
                                        "Processing exceeded the configured JOB_TIMEOUT_SECONDS limit. Increase it or set it to 0 and upload again.")
                        break
                if self.store.get(job["id"])["status"] == "processing":
                    self.store.fail(job["id"], "The processing worker stopped unexpectedly. Please try again.")
        except Exception:
            logger.exception("Could not run job %s", job["id"])
            self.store.fail(job["id"], "The server could not start processing. Please try again later.")
        finally:
            if process is not None:
                self.kill(process)
            self.remove_audio(job)
