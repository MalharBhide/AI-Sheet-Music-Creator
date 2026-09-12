import shutil
import subprocess
import wave
from types import SimpleNamespace

import pytest
import soundfile as sf

from app.config import Settings
from app.models import PipelineError
from app.services.audio_preprocess import normalize_audio

needs_ffmpeg = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is not installed')


def wav(path, seconds, *, channels=2, sample_rate=44100):
    with wave.open(str(path), 'wb') as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        remaining = round(sample_rate * seconds)
        while remaining:
            frames = min(remaining, sample_rate)
            file.writeframes(b'\x00\x00' * channels * frames)
            remaining -= frames


@needs_ffmpeg
def test_real_ffmpeg_decodes_to_model_format(tmp_path, settings):
    source, output = tmp_path / 'input.wav', tmp_path / 'work/input.wav'
    wav(source, 1)
    assert normalize_audio(source, output, settings) == 1
    with wave.open(str(output), 'rb') as file:
        assert file.getnchannels() == 1
        assert file.getframerate() == 22050


@needs_ffmpeg
def test_full_mix_decoding_preserves_stereo_for_separation(tmp_path, settings):
    import numpy as np

    source, output = tmp_path / 'stereo.wav', tmp_path / 'work/input.wav'
    samples = np.column_stack((np.full(44100, .1), np.full(44100, -.1)))
    sf.write(str(source), samples, 44100)
    assert normalize_audio(source, output, settings, preserve_stereo=True) == 1
    decoded, rate = sf.read(str(output))
    assert rate == 44100
    assert decoded.shape == (44100, 2)
    assert decoded[:, 0].mean() == pytest.approx(.1, abs=.0001)
    assert decoded[:, 1].mean() == pytest.approx(-.1, abs=.0001)


@needs_ffmpeg
def test_long_audio_is_rejected_instead_of_silently_truncated(tmp_path, settings):
    source, output = tmp_path / 'input.wav', tmp_path / 'work/input.wav'
    wav(source, 5)
    with pytest.raises(PipelineError, match='exceeds'):
        normalize_audio(source, output, settings)
    with wave.open(str(output), 'rb') as file:
        assert file.getnframes() / file.getframerate() <= settings.max_audio_seconds + 1


@needs_ffmpeg
def test_disguised_non_audio_is_rejected(tmp_path, settings):
    source = tmp_path / 'bad.mp3'
    source.write_bytes(b'This is not audio')
    with pytest.raises(PipelineError, match='could not be decoded'):
        normalize_audio(source, tmp_path / 'output.wav', settings)


@needs_ffmpeg
@pytest.mark.parametrize(('sample_rate', 'channels', 'encoding'), [
    (8000, 1, ['-b:a', '24k']),
    (11025, 2, ['-q:a', '7']),
    (16000, 1, ['-b:a', '32k']),
    (22050, 2, ['-q:a', '4']),
    (24000, 1, ['-b:a', '48k']),
    (32000, 2, ['-b:a', '128k']),
    (44100, 2, ['-q:a', '2']),
    (48000, 1, ['-b:a', '320k']),
])
def test_mp3_variants_decode_regardless_of_filename(tmp_path, settings, sample_rate,
                                                   channels, encoding):
    source = tmp_path / 'source.wav'
    wav(source, .125, channels=channels, sample_rate=sample_rate)
    # Include ID3 metadata, CBR and VBR, MPEG-1/2/2.5, and a misleading suffix.
    encoded = tmp_path / 'audio.bin'
    subprocess.run([settings.ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(source), '-c:a', 'libmp3lame', *encoding,
                    '-metadata', 'title=Short recording', '-f', 'mp3', str(encoded)],
                   check=True, capture_output=True)
    output = tmp_path / 'normalized.wav'
    duration = normalize_audio(encoded, output, settings)
    assert duration == pytest.approx(.125, abs=.001)
    info = sf.info(str(output))
    assert (info.channels, info.samplerate, info.subtype) == (1, 22050, 'PCM_16')


@needs_ffmpeg
def test_default_settings_decode_beyond_former_maximum_without_truncation(tmp_path):
    source, output = tmp_path / 'long.wav', tmp_path / 'normalized.wav'
    wav(source, 601.125, channels=1, sample_rate=22050)
    settings = Settings(_env_file=None)
    assert settings.max_audio_seconds == 0
    assert normalize_audio(source, output, settings) == pytest.approx(601.125, abs=.001)


@needs_ffmpeg
def test_mp3_without_duration_header_is_fully_decoded(tmp_path):
    encoded = tmp_path / 'stream.mp3'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=61.125', '-c:a', 'libmp3lame',
                    '-q:a', '5', '-write_xing', '0', str(encoded)], check=True,
                   capture_output=True)
    duration = normalize_audio(encoded, tmp_path / 'normalized.wav', Settings(_env_file=None))
    # Headerless MP3 includes encoder delay and frame padding; don't trust tags.
    assert 61.125 <= duration < 61.3


@needs_ffmpeg
def test_empty_audio_is_rejected(tmp_path, settings):
    source = tmp_path / 'empty.wav'
    wav(source, 0)
    with pytest.raises(PipelineError, match='audio samples'):
        normalize_audio(source, tmp_path / 'normalized.wav', settings)


def test_decoder_has_no_implicit_time_limit_and_reads_rf64(tmp_path, settings, monkeypatch):
    settings.max_audio_seconds = 0
    source, output = tmp_path / 'input.mp3', tmp_path / 'normalized.wav'

    def decode(command, **kwargs):
        assert '-t' not in command
        assert 'timeout' not in kwargs
        assert command[command.index('-rf64') + 1] == 'auto'
        sf.write(str(output), [0.0] * 2205, 22050, format='RF64', subtype='PCM_16')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('app.services.audio_preprocess.subprocess.run', decode)
    assert normalize_audio(source, output, settings) == .1
