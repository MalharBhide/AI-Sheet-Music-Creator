"""Opt-in integration gate: no substitutions for FFmpeg, Basic Pitch, music21 or MuseScore."""
import importlib.util
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from main import create_app, dependencies


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_PIPELINE_INTEGRATION') != '1', reason='Set RUN_PIPELINE_INTEGRATION=1 to run the complete native pipeline')
def test_audio_upload_to_all_download_formats(tmp_path):
    settings = Settings(storage_root=tmp_path / 'data', _env_file=None)
    assert all(dependencies(settings).values()), 'Install every native and Python dependency before running this integration gate.'
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'make_sample.py'
    spec = importlib.util.spec_from_file_location('make_sample', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / 'sample.wav'
    module.make_sample(audio)
    with TestClient(create_app(settings)) as client:
        result = client.post('/api/upload', files={'file': ('sample.wav', audio.read_bytes(), 'audio/wav')})
        assert result.status_code == 202, result.text
        url = result.headers['Location']
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            job = client.get(url).json()
            if job['status'] in ('done', 'failed'):
                break
            time.sleep(1)
        assert job['status'] == 'done', job
        expected = {'application/pdf', 'image/svg+xml', 'audio/midi', 'application/vnd.recordare.musicxml+xml', 'application/zip'}
        assert {a['media_type'] for a in job['artifacts']} == expected
        for artifact in job['artifacts']:
            response = client.get(artifact['url'])
            assert response.status_code == 200
            assert response.content
        assert not (settings.jobs_dir / job['job_id'] / 'upload/input.wav').exists()


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
