import shutil
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models import ScoreOptions
from app.models.job import JobResponse, job_response
from app.routes.dependencies import dependencies
from app.services.job_store import QueueFull
from app.services.storage import SUPPORTED_AUDIO_EXTENSIONS, display_filename

router = APIRouter(prefix='/api', tags=['uploads'])


@router.post('/upload', status_code=202, response_model=JobResponse)
async def upload_audio(
    request: Request,
    file: Annotated[UploadFile, File()],
    tempo_bpm: Annotated[float | None, Form(ge=30, le=240)] = None,
    time_signature: Annotated[Literal['4/4', '3/4', '6/8'], Form()] = '4/4',
    grid: Annotated[Literal['eighth', 'sixteenth'], Form()] = 'sixteenth',
):
    settings = request.app.state.settings
    store = request.app.state.store
    directory = None
    saved = False
    try:
        # Display names are shortened for storage/UI; determine the extension first
        # so a long (otherwise valid) MP3 filename does not lose its suffix.
        extension = Path((file.filename or '').replace('\\', '/')).suffix.lower()
        filename = display_filename(file.filename)
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(415, 'Choose a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF file.')
        try:
            options = ScoreOptions(tempo_bpm=tempo_bpm if tempo_bpm is not None else settings.default_tempo_bpm,
                                   time_signature=time_signature, grid=grid)
        except ValidationError as exc:
            raise HTTPException(422, 'Invalid score settings.') from exc
        checks = dependencies(settings)
        if not all(checks.values()):
            missing = ', '.join(key for key, ok in checks.items() if not ok)
            raise HTTPException(503, f'The server is missing required dependencies: {missing}. See setup instructions.')
        job_id = str(uuid4())
        directory = settings.jobs_dir / job_id
        (directory / 'upload').mkdir(parents=True)
        input_name = f'upload/input{extension}'
        total = 0
        with (directory / input_name).open('xb') as destination:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if settings.max_upload_mb and total > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f'Audio files must be {settings.max_upload_mb} MB or smaller.')
                destination.write(chunk)
        if total == 0:
            raise HTTPException(400, 'The uploaded file is empty.')
        try:
            job = store.create(job_id, filename, input_name, options, settings.max_queued_jobs)
        except QueueFull as exc:
            raise HTTPException(429, 'The queue is full. Please try again shortly.',
                                headers={'Retry-After': '30'}) from exc
        saved = True
        return JSONResponse(job_response(job).model_dump(mode='json'), status_code=202,
                            headers={'Location': f'/api/jobs/{job_id}'})
    finally:
        await file.close()
        if directory is not None and not saved:
            shutil.rmtree(directory, ignore_errors=True)
