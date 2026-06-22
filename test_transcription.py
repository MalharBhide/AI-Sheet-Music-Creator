from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.models.job import JobStatus  # noqa: E402
from app.services.job_store import job_store  # noqa: E402
from app.services.storage import get_job_paths  # noqa: E402
from app.workers.transcription_worker import run_transcription_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the audio-to-piano-score pipeline locally.")
    parser.add_argument("audio_file", type=Path)
    args = parser.parse_args()

    audio_file = args.audio_file.expanduser().resolve()
    if not audio_file.exists():
      print(f"Audio file not found: {audio_file}", file=sys.stderr)
      return 1

    job = job_store.create(audio_file.name)
    paths = get_job_paths(job.job_id, audio_file.name)
    paths.upload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audio_file, paths.original_audio)

    run_transcription_job(job.job_id)
    result = job_store.get(job.job_id)

    print(f"Job: {result.job_id}")
    print(f"Status: {result.status}")
    if result.status == JobStatus.failed:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    print(f"MIDI: {paths.midi}")
    print(f"MusicXML: {paths.musicxml}")
    print(f"PDF: {paths.pdf}")
    print(f"SVG: {paths.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

