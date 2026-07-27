from app.models.job import JobStatus
from app.services.audio_preprocess import convert_to_wav
from app.services.job_store import job_store
from app.services.midi_to_score import midi_to_piano_musicxml
from app.services.piano_transcription import transcribe_audio_to_piano_midi
from app.services.score_render import render_musicxml_to_pdf_and_svg
from app.services.storage import get_job_paths, relative_to_storage


def run_transcription_job(job_id: str) -> None:
    try:
        job = job_store.get(job_id)
        paths = get_job_paths(job_id, job.original_filename)

        job_store.update_status(job_id, JobStatus.preprocessing)
        convert_to_wav(paths.original_audio, paths.wav_audio)

        job_store.update_status(job_id, JobStatus.transcribing)
        transcribe_audio_to_piano_midi(paths.wav_audio, paths.midi)
        job_store.set_file(job_id, "midi", relative_to_storage(paths.midi))

        job_store.update_status(job_id, JobStatus.scoring)
        midi_to_piano_musicxml(
            paths.midi,
            paths.musicxml,
            title=f"Piano Score - {job.original_filename}",
        )
        job_store.set_file(job_id, "musicxml", relative_to_storage(paths.musicxml))

        job_store.update_status(job_id, JobStatus.rendering)
        pdf_path, svg_path = render_musicxml_to_pdf_and_svg(paths.musicxml, paths.pdf, paths.svg)
        job_store.set_file(job_id, "pdf", relative_to_storage(pdf_path))
        job_store.set_file(job_id, "svg", relative_to_storage(svg_path))

        job_store.update_status(job_id, JobStatus.done)
    except Exception as exc:
        job_store.fail(job_id, str(exc))

