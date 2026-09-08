from uuid import UUID

from fastapi import APIRouter, Request

from app.models.job import JobResponse, job_response
from app.routes.dependencies import lookup

router = APIRouter(prefix='/api/jobs', tags=['jobs'])


@router.get('/{job_id}', response_model=JobResponse)
def get_job(job_id: UUID, request: Request):
    return job_response(lookup(request, job_id))
