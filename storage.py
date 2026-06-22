from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from fastapi import UploadFile

from app.config import get_settings


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobPaths:
    job_dir: Path
    upload_dir: Path
    work_dir: Path
    output_dir: Path
    original_audio: Path
    wav_audio: Path
    midi: Path
    musicxml: Path
    pdf: Path
    svg: Path


def get_job_paths(job_id: str, original_filename: str | None = None) -> JobPaths:
    settings = get_settings()
    job_dir = settings.storage_root / "jobs" / job_id
    upload_dir = job_dir / "upload"
    work_dir = job_dir / "work"
    output_dir = job_dir / "outputs"

    if original_filename:
        original = upload_dir / _safe_filename(original_filename)
    else:
        existing_uploads = list(upload_dir.glob("*"))
        original = existing_uploads[0] if existing_uploads else upload_dir / "original.audio"

    return JobPaths(
        job_dir=job_dir,
        upload_dir=upload_dir,
        work_dir=work_dir,
        output_dir=output_dir,
        original_audio=original,
        wav_audio=work_dir / "input.wav",
        midi=output_dir / "transcription.mid",
        musicxml=output_dir / "score.musicxml",
        pdf=output_dir / "score.pdf",
        svg=output_dir / "score.svg",
    )


async def save_upload(job_id: str, upload_file: UploadFile) -> Path:
    if not upload_file.filename:
        raise StorageError("Uploaded file is missing a filename.")

    suffix = Path(upload_file.filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise StorageError(f"Unsupported audio type '{suffix}'. Supported types: {supported}.")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    paths = get_job_paths(job_id, upload_file.filename)
    paths.upload_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with paths.original_audio.open("wb") as destination:
        while chunk := await upload_file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise StorageError(f"Upload is larger than {settings.max_upload_mb} MB.")
            destination.write(chunk)

    return paths.original_audio


def relative_to_storage(path: Path) -> str:
    settings = get_settings()
    return str(path.resolve().relative_to(settings.storage_root.resolve()))


def resolve_stored_path(relative_path: str) -> Path:
    settings = get_settings()
    candidate = (settings.storage_root / relative_path).resolve()
    storage_root = settings.storage_root.resolve()
    try:
        candidate.relative_to(storage_root)
    except ValueError:
        raise StorageError("Invalid stored path.")
    return candidate


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem or "audio"
    suffix = Path(filename).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe_stem or 'audio'}{suffix}"
