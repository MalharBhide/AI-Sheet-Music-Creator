from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from app.models import PipelineError, ScoreOptions

CHUNK_SECONDS = 30
CONTEXT_SECONDS = 1
MINIMUM_INPUT_SECONDS = 1


def _audio_chunks(audio_path: Path, chunk_path: Path) -> Iterator[tuple[float, float, float]]:
    """Yield (read offset, core start, core end), keeping only one PCM chunk in RAM."""
    import numpy as np

    with sf.SoundFile(str(audio_path)) as audio:
        if audio.samplerate != 22050 or audio.channels != 1:
            raise PipelineError("Transcription needs audio normalized to mono at 22050 Hz.")
        if not audio.frames:
            raise PipelineError("The recording did not contain any audio samples.")
        chunk_frames = CHUNK_SECONDS * audio.samplerate
        context_frames = CONTEXT_SECONDS * audio.samplerate
        for core_start in range(0, audio.frames, chunk_frames):
            core_end = min(audio.frames, core_start + chunk_frames)
            read_start = max(0, core_start - context_frames)
            read_end = min(audio.frames, core_end + context_frames)
            audio.seek(read_start)
            samples = audio.read(read_end - read_start, dtype="float32")
            # Basic Pitch's frame postprocessing needs a nonempty time axis even
            # for sub-frame recordings. Later clipping removes all padded time.
            minimum_frames = MINIMUM_INPUT_SECONDS * audio.samplerate
            if len(samples) < minimum_frames:
                samples = np.pad(samples, (0, minimum_frames - len(samples)))
            sf.write(str(chunk_path), samples, audio.samplerate, subtype="PCM_16")
            yield (read_start / audio.samplerate, core_start / audio.samplerate,
                   core_end / audio.samplerate)


def _append_chunk_notes(instrument, chunk_midi, offset: float, core_start: float,
                        core_end: float, boundary_notes: dict) -> dict:
    """Keep each window's core once, joining sustained pitches across its boundary."""
    next_boundary = {}
    notes = sorted((note for part in chunk_midi.instruments for note in part.notes),
                   key=lambda note: (note.start, note.pitch, note.end))
    for note in notes:
        original_start = offset + note.start
        start = max(core_start, original_start)
        end = min(core_end, offset + note.end)
        if end <= start:
            continue  # Only context or model padding; another core owns this time.
        note.start, note.end = start, end
        previous = boundary_notes.get(note.pitch)
        if previous is not None and original_start < core_start and start == core_start:
            # A new onset at/after the seam remains a separate repeated note.
            previous.end = end
            current = previous
            boundary_notes.pop(note.pitch, None)
        else:
            instrument.notes.append(note)
            current = note
        if end == core_end:
            next_boundary[note.pitch] = current
    return next_boundary


def transcribe(audio_path: Path, midi_path: Path, options: ScoreOptions, *,
               progress_callback: Callable[[float], None] | None = None) -> None:
    # Heavy ML imports stay in the isolated child process, never in the HTTP server.
    try:
        import pretty_midi
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict
    except ImportError as exc:
        raise PipelineError("Transcription is unavailable. Install the backend transcription dependencies.") from exc

    model = Model(ICASSP_2022_MODEL_PATH)
    midi = pretty_midi.PrettyMIDI(initial_tempo=options.tempo_bpm)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    midi.instruments.append(piano)
    duration = sf.info(str(audio_path)).duration
    boundary_notes = {}
    with TemporaryDirectory(prefix="transcription-", dir=audio_path.parent) as temporary:
        chunk_path = Path(temporary) / "chunk.wav"
        for offset, core_start, core_end in _audio_chunks(audio_path, chunk_path):
            model_output, chunk_midi, note_events = predict(
                str(chunk_path),
                model_or_model_path=model,
                minimum_frequency=27.5,  # A0, bottom of a standard piano.
                maximum_frequency=4186.01,  # C8.
                minimum_note_length=100.0,
                multiple_pitch_bends=False,
                midi_tempo=options.tempo_bpm,
            )
            boundary_notes = _append_chunk_notes(piano, chunk_midi, offset, core_start,
                                                 core_end, boundary_notes)
            # Dense model arrays are the expensive part; never accumulate them.
            del model_output, chunk_midi, note_events
            if progress_callback:
                progress_callback(min(1.0, core_end / duration))
    # Silence is a valid recording: write an empty MIDI and let notation create rests.
    piano.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_path))
