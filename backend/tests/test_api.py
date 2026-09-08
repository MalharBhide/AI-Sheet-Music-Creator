from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main import create_app


def upload(client, content=b"test audio bytes", filename="recording.wav", **data):
    return client.post("/api/upload", files={"file": (filename, content, "audio/wav")}, data=data)


def test_upload_poll_and_persist(client, settings):
    result = upload(client, filename="../../my song.wav", tempo_bpm="96", grid="eighth")
    assert result.status_code == 202
    job = result.json()
    assert job["status"] == "queued"
    assert job["original_filename"] == "my song.wav"
    assert job["options"]["tempo_bpm"] == 96
    assert job["artifacts"] == []
    assert (settings.jobs_dir / job["job_id"] / "upload/input.wav").read_bytes() == b"test audio bytes"
    assert client.get(result.headers["Location"]).json()["job_id"] == job["job_id"]
    with TestClient(create_app(settings, start_worker=False)) as restarted:
        assert restarted.get(result.headers["Location"]).json()["status"] == "queued"


@pytest.mark.parametrize(("content", "filename", "status"), [
    (b"", "empty.wav", 400), (b"x", "bad.exe", 415),
    (b"x" * (1024 * 1024 + 1), "big.wav", 413),
    (b"x" * (2 * 1024 * 1024), "big.wav", 413),
])
def test_invalid_uploads_leave_no_job_files(client, settings, content, filename, status):
    assert upload(client, content, filename).status_code == status
    assert list(settings.jobs_dir.iterdir()) == []


@pytest.mark.parametrize("data", [{"tempo_bpm": "0"}, {"tempo_bpm": "nan"},
                                  {"time_signature": "5/0"}, {"grid": "invalid"}])
def test_settings_validation(client, data):
    assert upload(client, **data).status_code == 422


def test_queue_capacity_and_retry_header(client, settings):
    assert upload(client).status_code == 202
    assert upload(client).status_code == 202
    result = upload(client)
    assert result.status_code == 429
    assert result.headers["Retry-After"] == "30"
    assert len(list(settings.jobs_dir.iterdir())) == 2


def test_missing_dependencies(client, monkeypatch):
    monkeypatch.setattr("main.dependencies", lambda _: {"musescore": False})
    monkeypatch.setattr("app.routes.uploads.dependencies", lambda _: {"musescore": False})
    assert client.get("/api/health").json()["ready"] is False
    assert upload(client).status_code == 503


def test_downloads_are_only_for_completed_manifest_files(client, settings):
    job = upload(client).json()
    root = f"/api/jobs/{job['job_id']}"
    assert client.get(f"{root}/files/score.pdf").status_code == 409
    store = client.app.state.store
    artifact = {"name": "score.pdf", "label": "PDF", "media_type": "application/pdf"}
    store.update(job["job_id"], status="completed", stage="completed", progress=100, artifacts=[artifact])
    (settings.jobs_dir / job["job_id"] / "outputs").mkdir()
    (settings.jobs_dir / job["job_id"] / "outputs/score.pdf").write_bytes(b"%PDF-1.4\nfixture")
    result = client.get(f"{root}/files/score.pdf")
    assert result.status_code == 200
    assert result.content.startswith(b"%PDF-")
    completed = client.get(root).json()
    assert completed['status'] == 'done'
    assert completed['download_urls']['pdf'] == f"/api/downloads/{job['job_id']}/pdf"
    assert client.get(completed['download_urls']['pdf']).content == result.content
    assert client.get(f"/api/downloads/{job['job_id']}/unknown").status_code == 404
    assert "attachment" in result.headers["Content-Disposition"]
    assert "inline" in client.get(f"{root}/files/score.pdf?preview=true").headers["Content-Disposition"]
    assert client.get(f"{root}/files/input.wav").status_code == 404
    assert client.get(f"{root}/files/worker.log").status_code == 404
    assert client.get(f"{root}/files/..%5Cjobs.sqlite3").status_code == 404
    assert client.get(f"/api/jobs/{uuid4()}").status_code == 404
    assert client.get("/api/jobs/not-a-uuid").status_code == 422


def test_body_limit_without_content_length(client):
    boundary = "piano-boundary"
    header = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    def chunks():
        yield header
        for _ in range(20):
            yield b"x" * 65536
        yield f'\r\n--{boundary}--\r\n'.encode()
    response = client.post("/api/upload", content=chunks(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
