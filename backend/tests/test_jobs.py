import time
from uuid import uuid4

from app.models import ScoreOptions
from app.workers.job_runner import JobRunner
from app.services.job_store import JobStore
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
