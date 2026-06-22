from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.job_store import JobNotFoundError, job_store
from app.services.storage import resolve_stored_path


router = APIRouter(prefix="/api/downloads", tags=["downloads"])

CONTENT_TYPES = {
    "midi": "audio/midi",
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}

FILENAMES = {
    "midi": "piano-transcription.mid",
    "musicxml": "piano-score.musicxml",
    "pdf": "piano-score.pdf",
    "svg": "piano-score.svg",
}


@router.get("/{job_id}/{kind}")
def download_file(job_id: str, kind: str) -> FileResponse:
    if kind not in CONTENT_TYPES:
        raise HTTPException(status_code=404, detail="Unknown download type.")

    try:
        job = job_store.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found.") from None

    relative_path = getattr(job.files, kind)
    if not relative_path:
        raise HTTPException(status_code=404, detail=f"{kind} is not available yet.")

    path = resolve_stored_path(relative_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{kind} file is missing.")

    return FileResponse(
        path=path,
        media_type=CONTENT_TYPES[kind],
        filename=FILENAMES[kind],
    )

