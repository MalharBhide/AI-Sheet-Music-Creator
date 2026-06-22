from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    preprocessing = "preprocessing"
    transcribing = "transcribing"
    scoring = "scoring"
    rendering = "rendering"
    done = "done"
    failed = "failed"


class GeneratedFiles(BaseModel):
    midi: str | None = None
    musicxml: str | None = None
    pdf: str | None = None
    svg: str | None = None


class JobRecord(BaseModel):
    job_id: str
    original_filename: str
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    files: GeneratedFiles = Field(default_factory=GeneratedFiles)


class JobResponse(JobRecord):
    download_urls: GeneratedFiles = Field(default_factory=GeneratedFiles)

