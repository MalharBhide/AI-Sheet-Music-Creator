from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.models.job import JobResponse
from app.services.job_store import job_store
from app.services.storage import StorageError, save_upload
from app.workers.transcription_worker import run_transcription_job


router = APIRouter(prefix="/api", tags=["uploads"])


@router.post("/upload", response_model=JobResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JobResponse:
    job = job_store.create(file.filename or "audio")

    try:
        await save_upload(job.job_id, file)
    except StorageError as exc:
        job_store.fail(job.job_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(run_transcription_job, job.job_id)
    return job_store.response(job.job_id)

