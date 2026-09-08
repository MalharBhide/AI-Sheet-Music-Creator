from pathlib import Path
from uuid import UUID

from app.config import Settings

SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.aif', '.aiff'}


def display_filename(filename: str | None) -> str:
    name = Path((filename or 'recording').replace('\\', '/')).name
    return ''.join(char for char in name if char.isprintable())[:160]


def output_path(settings: Settings, job_id: UUID, name: str) -> Path:
    if Path(name).name != name or '\\' in name:
        raise ValueError('Invalid artifact name')
    return settings.jobs_dir / str(job_id) / 'outputs' / name
