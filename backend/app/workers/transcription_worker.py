import logging
import sys
from pathlib import Path
from uuid import UUID

from app.config import Settings
from app.models import PipelineError, ScoreOptions
from app.services.audio_preprocess import normalize_audio
from app.services.job_store import JobStore
from app.services.piano_transcription import transcribe
from app.services.score_render import render_score


def process_job(job_id: str, settings: Settings) -> None:
    store = JobStore(settings.database_path)
    job = store.get(job_id)
    if job is None or job["status"] != "processing":
        raise RuntimeError("Job must be claimed before processing")
    directory = settings.jobs_dir / job_id
    options = ScoreOptions.model_validate(job["options"])
    output = directory / "outputs"
    output.mkdir(exist_ok=True)
    (directory / "work").mkdir(exist_ok=True)
    try:
        store.update(job_id, stage="decoding", progress=10)
        duration = normalize_audio(directory / job["input_name"], directory / "work/input.wav", settings)
        store.update(job_id, stage="transcribing", progress=25)
        last_progress = 25

        def transcription_progress(fraction: float) -> None:
            nonlocal last_progress
            progress = min(64, 25 + int(39 * fraction))
            if progress != last_progress:
                store.update(job_id, progress=progress)
                last_progress = progress

        transcribe(directory / "work/input.wav", output / "transcription.mid", options,
                   progress_callback=transcription_progress)
        artifacts = [{"name": "transcription.mid", "label": "MIDI", "media_type": "audio/midi"}]
        store.update(job_id, stage="notating", progress=65, artifacts=artifacts)
        from app.services.midi_to_score import midi_to_musicxml

        midi_to_musicxml(output / "transcription.mid", output / "score.musicxml",
                         options, Path(job["filename"]).stem, duration_seconds=duration)
        artifacts.append({"name": "score.musicxml", "label": "MusicXML",
                          "media_type": "application/vnd.recordare.musicxml+xml"})
        store.update(job_id, stage="rendering", progress=85, artifacts=artifacts)
        artifacts = render_score(output / "score.musicxml", output, settings)
        store.update(job_id, status="completed", stage="completed", progress=100, artifacts=artifacts)
    except PipelineError as exc:
        store.fail(job_id, str(exc))
    except Exception:
        logging.exception("Unexpected processing failure for job %s", job_id)
        store.fail(job_id, "Processing failed. Details are in this job's server worker.log; please retry after resolving the logged error.")
    finally:
        # Original and decoded audio are not retained after processing.
        (directory / job["input_name"]).unlink(missing_ok=True)
        (directory / "work/input.wav").unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_job(str(UUID(sys.argv[1])), Settings())
