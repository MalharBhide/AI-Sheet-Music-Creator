"""Opt-in integration gate: no substitutions for FFmpeg, Basic Pitch, music21 or MuseScore."""
import importlib.util
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from main import create_app, dependencies


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_PIPELINE_INTEGRATION') != '1', reason='Set RUN_PIPELINE_INTEGRATION=1 to run the complete native pipeline')
@pytest.mark.parametrize('encoding', ['wav', 'mp3-vbr-long', 'mp3-short', 'mp3-silent', 'full-mix-mp3'])
def test_audio_upload_to_all_download_formats(tmp_path, encoding):
    settings = Settings(storage_root=tmp_path / 'data', _env_file=None)
    assert all(dependencies(settings).values()), 'Install every native and Python dependency before running this integration gate.'
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'make_sample.py'
    spec = importlib.util.spec_from_file_location('make_sample', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / 'sample.wav'
    module.make_sample(audio)
    if encoding == 'full-mix-mp3':
        audio.write_bytes((Path(__file__).resolve().parents[1] / 'app/assets/piano-study.wav').read_bytes())
    if encoding != 'wav':
        mp3 = tmp_path / 'sample.mp3'
        # Exercise the former 180-second failure through the real API and every
        # native stage. Sparse piano avoids turning this into a quality benchmark.
        filters = [] if encoding == 'full-mix-mp3' else ['-t', '0.125']
        if encoding == 'mp3-vbr-long':
            filters = ['-af', 'apad=whole_dur=181.125']
        if encoding == 'mp3-silent':
            filters = ['-af', 'volume=0']
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(audio), *filters,
                        '-ar', '48000', '-ac', '2', '-c:a', 'libmp3lame', '-q:a', '4',
                        str(mp3)], check=True)
        audio = mp3
    with TestClient(create_app(settings)) as client:
        result = client.post('/api/upload', files={'file': (audio.name, audio.read_bytes(), 'application/octet-stream')},
                             data={'tempo_bpm': '96' if encoding == 'full-mix-mp3' else '120',
                                   'transcription_mode': 'full_mix' if encoding == 'full-mix-mp3' else 'piano'})
        assert result.status_code == 202, result.text
        url = result.headers['Location']
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            job = client.get(url).json()
            if job['status'] in ('done', 'failed'):
                break
            time.sleep(1)
        assert job['status'] == 'done', job
        if encoding == 'full-mix-mp3':
            assert job['analysis']['engine'].startswith('Demucs')
            assert job['analysis']['note_count'] > 0
        expected = {'application/pdf', 'image/svg+xml', 'audio/midi', 'application/vnd.recordare.musicxml+xml', 'application/zip', 'application/json'}
        assert {a['media_type'] for a in job['artifacts']} == expected
        if encoding == 'mp3-silent':
            from xml.etree import ElementTree

            xml = ElementTree.parse(settings.jobs_dir / job['job_id'] / 'outputs/score.musicxml')
            assert not xml.findall('.//pitch')
            assert xml.findall('.//rest')
        if encoding == 'mp3-vbr-long':
            from music21 import converter

            score = converter.parse(str(settings.jobs_dir / job['job_id'] / 'outputs/score.musicxml'))
            assert score.highestTime >= 181.125 * 2  # Full duration at 120 BPM.
        for artifact in job['artifacts']:
            response = client.get(artifact['url'])
            assert response.status_code == 200
            assert response.content
        assert not list((settings.jobs_dir / job['job_id'] / 'upload').iterdir())
        assert not (settings.jobs_dir / job['job_id'] / 'work').exists()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_PIPELINE_INTEGRATION') != '1', reason='Set RUN_PIPELINE_INTEGRATION=1 to run the native renderer')
def test_native_renderer_preserves_multiple_pages(tmp_path):
    from music21 import note, stream
    from pypdf import PdfReader

    from app.models import ScoreOptions
    from app.services.midi_to_score import midi_to_musicxml
    from app.services.score_render import render_score

    settings = Settings(storage_root=tmp_path, _env_file=None)
    assert settings.renderer(), 'MuseScore is required'
    score = stream.Stream()
    for index in range(640):
        score.append(note.Note(60 + index % 12, quarterLength=.5))
    midi = tmp_path / 'transcription.mid'
    score.write('midi', fp=str(midi))
    xml = tmp_path / 'score.musicxml'
    midi_to_musicxml(midi, xml, ScoreOptions(), 'Multi-page integration test')
    artifacts = render_score(xml, tmp_path, settings)
    page_count = len(PdfReader(tmp_path / 'score.pdf').pages)
    assert page_count > 1
    assert sum(a['media_type'] == 'image/svg+xml' for a in artifacts) == page_count
