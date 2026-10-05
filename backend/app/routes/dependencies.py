import importlib.util
import shutil
from uuid import UUID

from fastapi import HTTPException, Request

from app.config import Settings
from app.services.accompaniment_verifier import (
    BOUNDARY_CHECKPOINT,
    BOUNDARY_GUARDIAN_CHECKPOINT,
    BROAD_CHECKPOINT,
    BROAD_GUARDIAN_CHECKPOINT,
    CONTEXT_CHECKPOINTS,
    LEFT_GUARDIAN_CHECKPOINT,
    LEFT_HAND_CHECKPOINT,
    LEFT_REFINEMENT_CHECKPOINT,
    LEFT_RELATIONS_CHECKPOINT,
    RESIDUAL_CHECKPOINT,
)
from app.services.accompaniment_verifier import CHECKPOINT as ACCOMPANIMENT_CHECKPOINT
from app.services.bass_articulation import CHECKPOINTS as BASS_ARTICULATION_CHECKPOINTS
from app.services.bass_harmonic import CHECKPOINTS as BASS_HARMONIC_CHECKPOINTS
from app.services.bass_residual import CHECKPOINTS as BASS_RESIDUAL_CHECKPOINTS
from app.services.bass_texture import CHECKPOINTS as BASS_TEXTURE_CHECKPOINTS
from app.services.bass_verifier import CHECKPOINTS as BASS_CHECKPOINTS
from app.services.vocal_melody import CHECKPOINT


def dependencies(settings: Settings) -> dict[str, bool]:
    return {'ffmpeg': bool(shutil.which(settings.ffmpeg_bin)),
            'basic_pitch': importlib.util.find_spec('basic_pitch') is not None,
            'piano_transcription': importlib.util.find_spec('piano_transcription_inference') is not None,
            'piano_model': settings.piano_model_path.is_file(),
            'vocal_model': CHECKPOINT.is_file(),
            'accompaniment_model': ACCOMPANIMENT_CHECKPOINT.is_file(),
            'accompaniment_context_models': all(path.is_file() for path, _ in CONTEXT_CHECKPOINTS),
            'accompaniment_residual_model': RESIDUAL_CHECKPOINT.is_file(),
            'accompaniment_left_hand_model': LEFT_HAND_CHECKPOINT.is_file(),
            'accompaniment_left_refinement_model': LEFT_REFINEMENT_CHECKPOINT.is_file(),
            'accompaniment_left_consensus_models': LEFT_RELATIONS_CHECKPOINT.is_file() and LEFT_GUARDIAN_CHECKPOINT.is_file(),
            'accompaniment_pitch_preserving_models': BROAD_CHECKPOINT.is_file() and BROAD_GUARDIAN_CHECKPOINT.is_file(),
            'accompaniment_repeat_attack_models': BOUNDARY_CHECKPOINT.is_file() and BOUNDARY_GUARDIAN_CHECKPOINT.is_file(),
            'bass_consensus_models': all(path.is_file() for path, _, _ in BASS_CHECKPOINTS),
            'bass_articulation_models': all(path.is_file() for path, _, _ in BASS_ARTICULATION_CHECKPOINTS),
            'bass_residual_models': all(path.is_file() for path, _, _, _ in BASS_RESIDUAL_CHECKPOINTS),
            'bass_texture_models': all(path.is_file() for path, _, _, _ in BASS_TEXTURE_CHECKPOINTS),
            'bass_harmonic_models': all(path.is_file() for path, _, _, _ in BASS_HARMONIC_CHECKPOINTS),
            'source_separation': importlib.util.find_spec('demucs') is not None,
            'music21': importlib.util.find_spec('music21') is not None,
            'musescore': settings.renderer() is not None}


def lookup(request: Request, job_id: UUID) -> dict:
    job = request.app.state.store.get(str(job_id))
    if job is None:
        raise HTTPException(404, 'This job was not found or has expired.')
    return job
