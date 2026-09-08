import shutil
import wave

import pytest

from app.models import PipelineError
from app.services.audio_preprocess import normalize_audio

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is not installed')


def wav(path, seconds):
    with wave.open(str(path), 'wb') as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(44100)
        file.writeframes(b'\x00\x00\x00\x00' * int(44100 * seconds))


def test_real_ffmpeg_decodes_to_model_format(tmp_path, settings):
    source, output = tmp_path / 'input.wav', tmp_path / 'work/input.wav'
    wav(source, 1)
    assert normalize_audio(source, output, settings) == 1
    with wave.open(str(output), 'rb') as file:
        assert file.getnchannels() == 1
        assert file.getframerate() == 22050


def test_long_audio_is_rejected_instead_of_silently_truncated(tmp_path, settings):
    source, output = tmp_path / 'input.wav', tmp_path / 'work/input.wav'
    wav(source, 5)
    with pytest.raises(PipelineError, match='exceeds'):
        normalize_audio(source, output, settings)
    with wave.open(str(output), 'rb') as file:
        assert file.getnframes() / file.getframerate() <= settings.max_audio_seconds + 1


def test_disguised_non_audio_is_rejected(tmp_path, settings):
    source = tmp_path / 'bad.mp3'
    source.write_bytes(b'This is not audio')
    with pytest.raises(PipelineError, match='could not be decoded'):
        normalize_audio(source, tmp_path / 'output.wav', settings)
