from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from app.models import PipelineError, ScoreOptions
from app.services.audio_analysis import (
    arrange_melody_register,
    balance_piano_arrangement,
    clean_notes,
    estimate_grid_phase,
    estimate_key,
    estimate_tempo,
    preserve_bass_releases,
    preserve_melody_releases,
    reduce_accompaniment,
)

CHUNK_SECONDS = 30
CONTEXT_SECONDS = 1
MINIMUM_INPUT_SECONDS = 1


def _audio_chunks(audio_path: Path, chunk_path: Path, *,
                  allow_stereo: bool = False) -> Iterator[tuple[float, float, float]]:
    """Yield (read offset, core start, core end), keeping only one PCM chunk in RAM."""
    import numpy as np

    with sf.SoundFile(str(audio_path)) as audio:
        normalized = audio.samplerate == 22050 and audio.channels == 1
        stereo_source = allow_stereo and audio.samplerate == 44100 and audio.channels == 2
        if not normalized and not stereo_source:
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
                padding = ((0, minimum_frames - len(samples)), (0, 0)) if samples.ndim == 2 else (0, minimum_frames - len(samples))
                samples = np.pad(samples, padding)
            sf.write(str(chunk_path), samples, audio.samplerate, subtype="PCM_16")
            yield (read_start / audio.samplerate, core_start / audio.samplerate,
                   core_end / audio.samplerate)


def _append_chunk_notes(instrument, chunk_midi, offset: float, core_start: float,
                        core_end: float, boundary_notes: dict) -> dict:
    """Keep each window's core once, joining sustained pitches across its boundary."""
    next_boundary = {}
    for previous in boundary_notes.values():
        # The preceding window already heard this tail in its right context.
        # Preserve it if the next window misses the continuation, but never
        # extend beyond the new core (which may be the recording's final core).
        previous.end = min(core_end, max(previous.end, getattr(previous, '_context_end', previous.end)))
    notes = sorted((note for part in chunk_midi.instruments for note in part.notes),
                   key=lambda note: (note.start, note.pitch, note.end))
    for note in notes:
        original_start = offset + note.start
        original_end = offset + note.end
        start = max(core_start, original_start)
        end = min(core_end, offset + note.end)
        if end <= start:
            continue  # Only context or model padding; another core owns this time.
        note.start, note.end = start, end
        previous = boundary_notes.get(note.pitch)
        if previous is not None and original_start < core_start and start == core_start:
            # A new onset at/after the seam remains a separate repeated note.
            previous.end = max(previous.end, end)
            current = previous
            boundary_notes.pop(note.pitch, None)
        else:
            if previous is not None and previous.end > start:
                previous.end = start  # Preserve an actual new attack on this key.
            instrument.notes.append(note)
            current = note
        if end == core_end:
            current._context_end = min(original_end, core_end + CONTEXT_SECONDS)
            next_boundary[note.pitch] = current
    return next_boundary


class _PianoEngine:
    """Dedicated onset/offset/velocity model trained on piano, with CPU inference."""
    name = "High-resolution piano CRNN"

    def __init__(self, checkpoint: Path | None, detail: str):
        if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size < 160_000_000:
            raise PipelineError("The high-resolution piano model is not installed. Run the server model setup and configure PIANO_MODEL_PATH, then retry.")
        try:
            import torch
            from piano_transcription_inference import PianoTranscription
        except ImportError as exc:
            raise PipelineError("Piano transcription needs the high-resolution piano dependencies. Install the backend transcription extra and retry.") from exc
        try:
            torch.set_num_threads(min(4, torch.get_num_threads()))
            # A checked path prevents upstream from shelling out to wget and
            # silently downloading weights into the process user's home folder.
            self.model = PianoTranscription(device="cpu", checkpoint_path=str(checkpoint))
            self.model.onset_threshold = 0.3 if detail == "balanced" else 0.25
        except Exception as exc:
            raise PipelineError("The piano model could not be loaded. Reinstall its checkpoint and check the server worker log.") from exc

    def predict(self, path: Path, role: str, bpm: float):
        import numpy as np
        import pretty_midi
        from scipy.signal import resample_poly

        samples, rate = sf.read(str(path), dtype="float32")
        samples = resample_poly(samples, 320, 441) if rate == 22050 else samples
        # Onset regression needs frames on both sides of an attack. A file that
        # begins immediately with a chord otherwise loses its opening notes.
        # Supply real silence as left context, then remove it from event times.
        prefix_seconds = 0.25
        samples = np.pad(samples, (int(16000 * prefix_seconds), 0))
        result = self.model.transcribe(samples, None)
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        piano = pretty_midi.Instrument(program=0)
        for event in result["est_note_events"]:
            start = max(0, float(event["onset_time"]) - prefix_seconds)
            end = float(event["offset_time"]) - prefix_seconds
            if end > start and 21 <= int(event["midi_note"]) <= 108:
                piano.notes.append(pretty_midi.Note(velocity=max(1, min(127, int(event["velocity"]))),
                                                   pitch=int(event["midi_note"]),
                                                   start=start, end=end))
        midi.instruments.append(piano)
        return midi


class _GeneralEngine:
    name = "Basic Pitch on isolated sources"

    def __init__(self, detail: str):
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict
        except ImportError as exc:
            raise PipelineError("Transcription is unavailable. Install the backend transcription dependencies.") from exc
        self.model = Model(ICASSP_2022_MODEL_PATH)
        self.predict_function = predict
        self.detail = detail
        self.vocal_model = None
        self.note_verifier = None
        self.verification = None

    def predict(self, path: Path, role: str, bpm: float):
        import pretty_midi

        limits = {"bass": (21, 60), "vocals": (45, 96), "melody": (45, 96),
                  "other": (36, 96)}
        low, high = limits[role]
        verify = role == 'other' and self.detail == 'balanced'
        arrays, midi, events = self.predict_function(
            str(path), model_or_model_path=self.model,
            # Basic Pitch zeros out constrained bands in its returned arrays.
            # Retain full evidence for the verifier, exactly as during training,
            # and apply the arrangement's pitch bounds to events afterwards.
            minimum_frequency=None if verify else float(pretty_midi.note_number_to_hz(low)),
            maximum_frequency=None if verify else float(pretty_midi.note_number_to_hz(high)),
            onset_threshold=0.5 if self.detail == "balanced" else 0.4,
            frame_threshold=0.3 if self.detail == "balanced" else 0.25,
            minimum_note_length=90.0 if self.detail == "balanced" else 60.0,
            multiple_pitch_bends=False, melodia_trick=self.detail == "detailed", midi_tempo=bpm,
        )
        if role == "vocals":
            from app.services.vocal_melody import VocalMelody

            if self.vocal_model is None:
                self.vocal_model = VocalMelody()
            # Decode continuous evidence, including frames for which Basic
            # Pitch's generic event thresholds emitted no note at all.
            return self.vocal_model.predict(path, arrays, bpm)
        if verify:
            from app.services.accompaniment_verifier import AccompanimentVerifier

            if self.note_verifier is None:
                self.note_verifier = AccompanimentVerifier()
                self.verification = {'model': self.note_verifier.name,
                                     'window_candidates': 0, 'window_rejections': 0}
            for part in midi.instruments:
                # Basic Pitch's maximum_frequency bound is exclusive.
                part.notes = [n for n in part.notes if low <= n.pitch < high]
            self.verification['window_candidates'] += sum(len(p.notes) for p in midi.instruments)
            self.verification['window_rejections'] += self.note_verifier.filter(path, arrays, midi)
        return midi


def transcribe(audio_path: Path, midi_path: Path, options: ScoreOptions, *,
               progress_callback: Callable[[float], None] | None = None,
               piano_model_path: Path | None = None) -> dict:
    """Separate full mixes before inference; use a dedicated model for solo piano.

    Full-mix output is a piano arrangement, not an exact original piano score.
    No unavailable engine is silently replaced with a different one.
    """
    import pretty_midi

    bpm, warnings = estimate_tempo(audio_path, options.tempo_bpm)
    mode, detail = options.transcription_mode, options.detail
    separator = None
    if mode == "piano":
        engine = _PianoEngine(piano_model_path, detail)
        roles = ["piano"]
    else:
        engine = _GeneralEngine(detail)
        roles = ["melody"]
        if mode == "full_mix":
            from app.services.source_separation import StemSeparator

            separator = StemSeparator()
            roles = ["vocals", "bass", "other"]
            warnings.append("Full-song mode makes a piano arrangement from separated vocals, bass and accompaniment. It cannot recover an exact original piano score; review overlapping instruments and missing notes.")
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    numerator, denominator = map(int, options.time_signature.split("/"))
    midi.time_signature_changes.append(pretty_midi.TimeSignature(numerator, denominator, 0))
    parts = {role: pretty_midi.Instrument(program=0, name=role.title()) for role in roles}
    boundaries = {role: {} for role in roles}
    duration = sf.info(str(audio_path)).duration
    with TemporaryDirectory(prefix="transcription-", dir=audio_path.parent) as temporary:
        directory = Path(temporary)
        chunk_path = directory / "chunk.wav"
        for offset, core_start, core_end in _audio_chunks(audio_path, chunk_path,
                                                        allow_stereo=mode == "full_mix"):
            import numpy as np

            samples, _ = sf.read(str(chunk_path), dtype="float32")
            has_signal = bool(np.max(np.abs(samples)) >= 1e-6)
            del samples
            sources = (separator.separate(chunk_path, directory) if separator else {roles[0]: chunk_path}) if has_signal else {}
            for role in roles:
                if role not in sources:
                    boundaries[role] = {}
                    continue
                chunk_midi = engine.predict(sources[role], role, bpm)
                boundaries[role] = _append_chunk_notes(parts[role], chunk_midi, offset,
                                                       core_start, core_end, boundaries[role])
                del chunk_midi
            if progress_callback:
                progress_callback(min(1.0, core_end / duration))

    raw_count = sum(len(part.notes) for part in parts.values())
    timing_offset = estimate_grid_phase([item for part in parts.values() for item in part.notes],
                                        tempo_bpm=bpm, grid=options.grid)
    retained = {}
    for role, part in parts.items():
        for item in part.notes:
            item.start = max(0, item.start - timing_offset)
            item.end = max(item.start, item.end - timing_offset)
        before = len(part.notes)
        if role == 'vocals' and options.grid == 'eighth':
            precise_count = len(clean_notes(part.notes, role=role, detail=detail,
                                            tempo_bpm=bpm, grid='sixteenth'))
        else:
            precise_count = 0
        part.notes = clean_notes(part.notes, role=role, detail=detail, tempo_bpm=bpm,
                                 grid=options.grid)
        if precise_count > len(part.notes) + max(2, before * .1):
            warnings.append('The simple eighth-note rhythm merges some fast melody notes. Choose Precise rhythm to retain more of the tune.')
        for item in part.notes:
            item.end = min(duration, item.end)
        part.notes = [item for item in part.notes if item.end > item.start]
        retained[role] = {'detected': before, 'score': len(part.notes)}
        midi.instruments.append(part)
    melody_shift = 0
    support_changes = 0
    bass_support_changes = 0
    if mode == "full_mix":
        melody_shift = arrange_melody_register(parts)
        # Reconcile duplicate low keys before choosing accompaniment voices,
        # so a redundant bass detection cannot occupy a backing voice slot.
        bass_support_changes = preserve_bass_releases(parts)
        parts['other'].notes = reduce_accompaniment(
            parts['other'].notes, detail=detail, melody=parts['vocals'].notes)
        support_changes = preserve_melody_releases(parts)
        balance_piano_arrangement(parts)
        for role, part in parts.items():
            retained[role]['score'] = len(part.notes)
    notes = [item for part in midi.instruments for item in part.notes]
    key_signature = estimate_key(notes)
    if not notes:
        warnings.append("No pitched notes were detected. Silence, percussion, very short clips or an unsuitable source may produce a score of rests.")
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_path))
    engine_name = f"Demucs htdemucs + {engine.name} + Vocadito melody decoder v1" if separator else engine.name
    verification = getattr(engine, 'verification', None)
    if verification:
        engine_name += f" + {verification['model']}"
    return {"engine": engine_name,
            "accompaniment_verification": verification,
            "tempo_bpm": bpm, "note_count": len(notes), "raw_note_count": raw_count,
            "key_signature": key_signature, "transcription_mode": mode,
            "timing_offset_seconds": timing_offset,
            "melody_octave_shift": melody_shift // 12, "notes_by_source": retained,
            "support_notes_changed_for_melody": support_changes,
            "support_notes_changed_for_bass": bass_support_changes,
            "duration_seconds": duration, "sources": list(parts), "warnings": warnings}
