from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from app.models.job import GeneratedFiles, JobRecord, JobResponse, JobStatus


class JobNotFoundError(KeyError):
    pass


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, original_filename: str) -> JobRecord:
        job = JobRecord(job_id=uuid4().hex, original_filename=original_filename)
        with self._lock:
            self._jobs[job.job_id] = job
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy(deep=True)

    def response(self, job_id: str) -> JobResponse:
        job = self.get(job_id)
        return JobResponse(**job.model_dump(), download_urls=_download_urls(job))

    def update_status(self, job_id: str, status: JobStatus) -> JobRecord:
        with self._lock:
            job = self._require(job_id)
            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            job.error = None
            return job.model_copy(deep=True)

    def set_file(self, job_id: str, kind: str, relative_path: str) -> JobRecord:
        with self._lock:
            job = self._require(job_id)
            setattr(job.files, kind, relative_path)
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def fail(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            job = self._require(job_id)
            job.status = JobStatus.failed
            job.error = error
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def _require(self, job_id: str) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job


def _download_urls(job: JobRecord) -> GeneratedFiles:
    urls = GeneratedFiles()
    for kind in ("midi", "musicxml", "pdf", "svg"):
        if getattr(job.files, kind):
            setattr(urls, kind, f"/api/downloads/{job.job_id}/{kind}")
    return urls


job_store = InMemoryJobStore()

