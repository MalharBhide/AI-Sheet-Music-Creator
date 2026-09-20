import importlib.util
import shutil
from uuid import UUID

from fastapi import HTTPException, Request

from app.config import Settings
from app.services.accompaniment_verifier import CHECKPOINT as ACCOMPANIMENT_CHECKPOINT
from app.services.vocal_melody import CHECKPOINT


def dependencies(settings: Settings) -> dict[str, bool]:
    return {'ffmpeg': bool(shutil.which(settings.ffmpeg_bin)),
            'basic_pitch': importlib.util.find_spec('basic_pitch') is not None,
            'piano_transcription': importlib.util.find_spec('piano_transcription_inference') is not None,
            'piano_model': settings.piano_model_path.is_file(),
            'vocal_model': CHECKPOINT.is_file(),
            'accompaniment_model': ACCOMPANIMENT_CHECKPOINT.is_file(),
            'source_separation': importlib.util.find_spec('demucs') is not None,
            'music21': importlib.util.find_spec('music21') is not None,
            'musescore': settings.renderer() is not None}


def lookup(request: Request, job_id: UUID) -> dict:
    job = request.app.state.store.get(str(job_id))
    if job is None:
        raise HTTPException(404, 'This job was not found or has expired.')
    return job
