from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

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
    midi: Optional[str] = None
    musicxml: Optional[str] = None
    pdf: Optional[str] = None
    svg: Optional[str] = None


class JobRecord(BaseModel):
    job_id: str
    original_filename: str
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    files: GeneratedFiles = Field(default_factory=GeneratedFiles)


class JobResponse(JobRecord):
    download_urls: GeneratedFiles = Field(default_factory=GeneratedFiles)

