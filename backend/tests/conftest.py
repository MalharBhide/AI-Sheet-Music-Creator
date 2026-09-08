import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_root=tmp_path / "data", max_upload_mb=1, max_audio_seconds=3,
                    max_queued_jobs=2, _env_file=None)


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr("app.routes.uploads.dependencies", lambda _: {"ffmpeg": True, "basic_pitch": True,
                                                                     "music21": True, "musescore": True})
    with TestClient(create_app(settings, start_worker=False)) as client:
        yield client
