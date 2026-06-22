from fastapi import APIRouter, HTTPException

from app.models.job import JobResponse
from app.services.job_store import JobNotFoundError, job_store


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return job_store.response(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found.") from None

