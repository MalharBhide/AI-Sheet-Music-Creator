from pathlib import Path

from app.models import PipelineError, ScoreOptions


def transcribe(audio_path: Path, midi_path: Path, options: ScoreOptions) -> None:
    # Heavy ML imports stay in the isolated child process, never in the HTTP server.
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise PipelineError("Transcription is unavailable. Install the backend transcription dependencies.") from exc

    _, midi, _ = predict(
        str(audio_path),
        minimum_frequency=27.5,  # A0, bottom of a standard piano.
        maximum_frequency=4186.01,  # C8.
        minimum_note_length=100.0,
        multiple_pitch_bends=False,
        midi_tempo=options.tempo_bpm,
    )
    if not any(instrument.notes for instrument in midi.instruments):
        raise PipelineError("No notes were detected. Try a clearer recording of a single instrument.")
    for instrument in midi.instruments:
        instrument.program = 0
        instrument.pitch_bends = []
    midi.write(str(midi_path))
