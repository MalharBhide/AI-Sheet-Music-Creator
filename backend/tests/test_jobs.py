import time
from uuid import uuid4

from app.models import ScoreOptions
from app.services.job_store import JobStore
from app.workers.job_runner import JobRunner
from app.workers.transcription_worker import process_job


def seed(settings):
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(settings.database_path)
    store.initialize()
    job_id = str(uuid4())
    directory = settings.jobs_dir / job_id
    (directory / 'upload').mkdir(parents=True)
    (directory / 'upload/input.wav').write_bytes(b'not an audio file')
    store.create(job_id, 'bad.wav', 'upload/input.wav', ScoreOptions(), capacity=2)
    return store, job_id, directory


def test_worker_records_failure_and_deletes_audio(settings):
    store, job_id, directory = seed(settings)
    assert store.claim_next()['id'] == job_id
    assert store.claim_next() is None
    process_job(job_id, settings)
    assert store.get(job_id)['status'] == 'failed'
    assert store.get(job_id)['error']
    assert not (directory / 'upload/input.wav').exists()
    assert not (directory / 'work/input.wav').exists()


def test_restart_recovery_and_retention(settings):
    store, job_id, directory = seed(settings)
    store.claim_next()
    store.recover_interrupted()
    assert store.get(job_id)['status'] == 'failed'
    assert 'restarted' in store.get(job_id)['error']
    with store.connect() as db:
        db.execute('UPDATE jobs SET updated_at = ? WHERE id = ?', (time.time() - 25 * 3600, job_id))
    JobRunner(settings, store).cleanup()
    assert store.get(job_id) is None
    assert not directory.exists()


def test_coordinator_runs_actual_child_process(settings):
    store, job_id, directory = seed(settings)
    runner = JobRunner(settings, store)
    runner.start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if store.get(job_id)['status'] == 'failed':
                break
            time.sleep(.1)
        assert store.get(job_id)['status'] == 'failed'
        assert 'decoded' in store.get(job_id)['error'] or 'FFmpeg' in store.get(job_id)['error']
    finally:
        runner.stop()
    assert not (directory / 'upload/input.wav').exists()


def test_late_render_failure_preserves_downloadable_intermediate_files(settings, monkeypatch):
    from app.models import PipelineError

    store, job_id, directory = seed(settings)
    store.claim_next()
    monkeypatch.setattr('app.workers.transcription_worker.normalize_audio', lambda *args: 200.0)

    def transcribe(audio, destination, options, *, progress_callback):
        destination.write_bytes(b'MThd')
        progress_callback(.5)
        assert store.get(job_id)['progress'] == 44
        progress_callback(1)

    def notate(source, destination, options, title, *, duration_seconds):
        assert duration_seconds == 200.0
        destination.write_text('<score-partwise/>')

    def render(*args):
        raise PipelineError('Renderer unavailable')

    monkeypatch.setattr('app.workers.transcription_worker.transcribe', transcribe)
    monkeypatch.setattr('app.services.midi_to_score.midi_to_musicxml', notate)
    monkeypatch.setattr('app.workers.transcription_worker.render_score', render)
    process_job(job_id, settings)
    job = store.get(job_id)
    assert job['status'] == 'failed'
    assert {a['name'] for a in job['artifacts']} == {'transcription.mid', 'score.musicxml'}
    assert (directory / 'outputs/transcription.mid').exists()
    assert (directory / 'outputs/score.musicxml').exists()
    assert not (directory / 'upload/input.wav').exists()


def test_default_settings_remove_recording_and_deadline_caps():
    from app.config import Settings

    settings = Settings(_env_file=None)
    assert settings.max_audio_seconds == settings.max_upload_mb == 0
    assert settings.job_timeout_seconds == settings.render_timeout_seconds == 0
    large = Settings(max_audio_seconds=86400, max_upload_mb=10000,
                     job_timeout_seconds=86400, render_timeout_seconds=7200, _env_file=None)
    assert large.max_audio_seconds == 86400


def test_disabled_job_deadline_allows_long_running_child(settings, monkeypatch):
    store, job_id, _ = seed(settings)
    job = store.claim_next()
    settings.job_timeout_seconds = 0
    runner = JobRunner(settings, store)
    monkeypatch.setattr(runner.stopping, 'wait', lambda _: False)
    monkeypatch.setattr('app.workers.job_runner.time.monotonic', lambda: 10**12)

    class Child:
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= 5:
                store.update(job_id, status='completed', stage='completed', progress=100)
                return 0
            return None

    monkeypatch.setattr('app.workers.job_runner.subprocess.Popen', lambda *args, **kwargs: Child())
    runner.execute(job)
    assert store.get(job_id)['status'] == 'completed'
