from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.pipeline import Artifact, ScoreOptions

JobStatus = Literal['queued', 'preprocessing', 'transcribing', 'scoring', 'rendering', 'done', 'failed']


class GeneratedFiles(BaseModel):
    midi: str | None = None
    musicxml: str | None = None
    pdf: str | None = None
    svg: str | None = None
    svg_zip: str | None = None
    playback: str | None = None


class JobResponse(BaseModel):
    job_id: str
    original_filename: str
    status: JobStatus
    stage: str
    progress: int
    created_at: datetime
    updated_at: datetime
    options: ScoreOptions
    error: str | None
    files: GeneratedFiles = Field(default_factory=GeneratedFiles)
    download_urls: GeneratedFiles = Field(default_factory=GeneratedFiles)
    svg_pages: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    analysis: dict = Field(default_factory=dict)


def job_response(job: dict) -> JobResponse:
    """Preserve the existing API field names and status vocabulary."""
    status = {'completed': 'done', 'queued': 'queued', 'failed': 'failed'}.get(job['status'])
    if status is None:
        status = {'transcribing': 'transcribing', 'notating': 'scoring', 'rendering': 'rendering'}.get(job['stage'], 'preprocessing')
    names = {'midi': 'transcription.mid', 'musicxml': 'score.musicxml', 'pdf': 'score.pdf',
             'svg': 'page-1.svg', 'svg_zip': 'score-svgs.zip', 'playback': 'playback.json'}
    existing = {artifact['name'] for artifact in job['artifacts']}
    files = {kind: name for kind, name in names.items() if name in existing}
    urls = {kind: f"/api/downloads/{job['id']}/{kind}" for kind in files}
    artifacts = [{**a, 'url': f"/api/jobs/{job['id']}/files/{a['name']}"} for a in job['artifacts']]
    return JobResponse(job_id=job['id'], original_filename=job['filename'], status=status,
                       stage=job['stage'], progress=job['progress'], options=job['options'],
                       created_at=datetime.fromtimestamp(job['created_at'], UTC),
                       updated_at=datetime.fromtimestamp(job['updated_at'], UTC),
                       error=job['error'], files=GeneratedFiles(**files),
                       download_urls=GeneratedFiles(**urls), artifacts=artifacts,
                       analysis=job.get('analysis', {}),
                       svg_pages=[a['url'] for a in artifacts if a['media_type'] == 'image/svg+xml'])
