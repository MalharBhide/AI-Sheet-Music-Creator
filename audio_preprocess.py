from pathlib import Path
import shutil
import subprocess

from app.config import get_settings


class AudioPreprocessError(RuntimeError):
    pass


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 22050) -> Path:
    settings = get_settings()
    ffmpeg = _resolve_binary(settings.ffmpeg_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise AudioPreprocessError(f"ffmpeg could not convert the audio file. {detail}")

    return output_path


def _resolve_binary(binary_name: str) -> str:
    resolved = shutil.which(binary_name)
    if not resolved:
        raise AudioPreprocessError(
            f"Could not find '{binary_name}'. Install ffmpeg or set FFMPEG_BIN."
        )
    return resolved

