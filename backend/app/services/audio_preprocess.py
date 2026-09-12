import logging
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from app.config import Settings
from app.models import PipelineError

logger = logging.getLogger(__name__)


def normalize_audio(source: Path, destination: Path, settings: Settings, *,
                    preserve_stereo: bool = False) -> float:
    """Stream the complete recording to disk, trusting decoded frames over duration tags."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-protocol_whitelist", "file,pipe",
        "-format_whitelist", "wav,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,aac,aiff",
        "-i", str(source.resolve()),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        # Source separation benefits from stereo placement and full-bandwidth
        # input; individual transcription models still use mono22.05kHz.
        "-ac", "2" if preserve_stereo else "1", "-ar", "44100" if preserve_stereo else "22050",
    ]
    # An administrator can opt into a cap. Zero means no duration limit.
    if settings.max_audio_seconds:
        command.extend(["-t", str(settings.max_audio_seconds + 1)])
    # RF64 keeps recordings beyond the WAV 4 GiB limit readable without truncation.
    command.extend(["-c:a", "pcm_s16le", "-rf64", "auto", "-f", "wav", str(destination)])
    try:
        # A fixed decode timeout rejects valid long recordings. The worker owns any
        # configured job deadline. Spooling diagnostics avoids growing Python RAM.
        with tempfile.TemporaryFile() as diagnostics:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=diagnostics,
                                    check=False)
            if result.returncode != 0:
                diagnostics.seek(0, 2)
                diagnostics.seek(max(0, diagnostics.tell() - 4000))
                logger.warning("FFmpeg failed: %s", diagnostics.read().decode(errors="replace"))
                raise PipelineError(
                    "This file could not be decoded as audio. Try exporting it as WAV or MP3."
                )
    except FileNotFoundError as exc:
        raise PipelineError("Audio decoding is unavailable. The server needs FFmpeg.") from exc
    try:
        info = sf.info(str(destination))
        duration = info.frames / info.samplerate
    except (RuntimeError, OSError) as exc:
        raise PipelineError("The recording did not contain readable audio.") from exc
    if info.frames == 0:
        raise PipelineError("The recording did not contain any audio samples.")
    if settings.max_audio_seconds and duration > settings.max_audio_seconds:
        raise PipelineError(f"The recording exceeds the {settings.max_audio_seconds}-second limit. Upload a shorter clip.")
    return duration
