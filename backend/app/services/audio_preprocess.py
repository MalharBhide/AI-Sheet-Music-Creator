import logging
import subprocess
import wave
from pathlib import Path

from app.config import Settings
from app.models import PipelineError

logger = logging.getLogger(__name__)


def normalize_audio(source: Path, destination: Path, settings: Settings) -> float:
    """Decode a bounded amount of audio. Never trust extension, MIME type or duration tags."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run([
            settings.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "wav,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,aac,aiff",
            "-i", str(source),
            "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "22050",
            "-t", str(settings.max_audio_seconds + 1), "-c:a", "pcm_s16le", str(destination),
        ], capture_output=True, timeout=60, check=False)
    except FileNotFoundError as exc:
        raise PipelineError("Audio decoding is unavailable. The server needs FFmpeg.") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("The recording took too long to decode. Try a shorter file.") from exc
    if result.returncode != 0:
        logger.warning("FFmpeg failed: %s", result.stderr.decode(errors="replace")[-4000:])
        raise PipelineError("This file could not be decoded as audio. Try exporting it as WAV or MP3.")
    try:
        with wave.open(str(destination), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
    except (wave.Error, OSError) as exc:
        raise PipelineError("The recording did not contain readable audio.") from exc
    if duration < 0.25:
        raise PipelineError("The recording must contain at least a quarter second of audio.")
    if duration > settings.max_audio_seconds:
        raise PipelineError(f"The recording exceeds the {settings.max_audio_seconds}-second limit. Upload a shorter clip.")
    return duration
