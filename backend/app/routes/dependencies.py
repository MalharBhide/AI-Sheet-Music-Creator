import importlib.util
import shutil
from uuid import UUID

from fastapi import HTTPException, Request

from app.config import Settings


def dependencies(settings: Settings) -> dict[str, bool]:
    return {'ffmpeg': bool(shutil.which(settings.ffmpeg_bin)),
            'basic_pitch': importlib.util.find_spec('basic_pitch') is not None,
            'music21': importlib.util.find_spec('music21') is not None,
            'musescore': settings.renderer() is not None}


def lookup(request: Request, job_id: UUID) -> dict:
    job = request.app.state.store.get(str(job_id))
    if job is None:
        raise HTTPException(404, 'This job was not found or has expired.')
    return job
