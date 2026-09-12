"""Bounded audio analysis and musical cleanup shared by transcription engines."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def estimate_tempo(audio_path: Path, requested: float | None) -> tuple[float, list[str]]:
    """Estimate one stable score tempo from at most three 30 second excerpts.

    The score currently has a constant tempo. Sampling beginning, middle and end
    avoids allocating a spectrogram proportional to the length of a recording.
    Rubato and half/double-time interpretations still need a musician's review.
    """
    if requested is not None:
        return float(requested), []
    candidates = []
    with sf.SoundFile(str(audio_path)) as audio:
        window = 30 * audio.samplerate
        last_start = max(0, audio.frames - window)
        starts = sorted({0, last_start // 2, last_start})
        for start in starts:
            audio.seek(start)
            samples = audio.read(window, dtype="float32")
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            if len(samples) < 3 * audio.samplerate or np.max(np.abs(samples)) < 1e-4:
                continue
            import librosa

            envelope = librosa.onset.onset_strength(y=samples, sr=audio.samplerate,
                                                   hop_length=256)
            if not np.any(envelope):
                continue
            tempo, beats = librosa.beat.beat_track(onset_envelope=envelope,
                                                 sr=audio.samplerate, hop_length=256,
                                                 trim=False)
            bpm = float(np.asarray(tempo).reshape(-1)[0])
            if len(beats) >= 4 and 30 <= bpm <= 240:
                # The returned tempo is limited by the analysis frame size.
                # Fitting detected beat positions yields much finer timing than
                # carrying that quantization error through a long score.
                spacing = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
                fitted = 60 * audio.samplerate / (256 * spacing) if spacing > 0 else bpm
                if abs(fitted - bpm) < 0.08 * bpm:
                    bpm = fitted
                candidates.append(bpm)
    if not candidates:
        return 120.0, ["A steady tempo could not be detected; 120 BPM was used. Set the tempo manually for a better rhythmic result."]
    # Select an observed tempo instead of averaging incompatible beat levels.
    # For example, 90 and 180 BPM are an ambiguity; their median135 is neither.
    bpm = min(candidates, key=lambda candidate: (
        sum(abs(math.log(candidate / other)) for other in candidates), abs(candidate - 120)))
    # Snap tiny numerical offsets from an integer metronome, but retain genuine
    # fractional tempi so timing does not drift through a long recording.
    rounded = round(bpm)
    bpm = float(rounded) if abs(bpm - rounded) < 0.15 else round(bpm, 2)
    warnings = ["Tempo is estimated. Check the beat, time signature and rhythm, especially for rubato or swing."]
    if max(candidates) - min(candidates) > 12:
        warnings.append("Different sections suggest different tempi. This score uses one tempo; review tempo changes in a notation editor.")
    return bpm, warnings


def clean_notes(notes: list, *, role: str, detail: str, tempo_bpm: float,
                grid: str) -> list:
    """Remove detection glitches and make a conservative, playable reduction.

    Solo piano keeps polyphony. Song stems are treated according to their musical
    role: one vocal melody, one bass line, and a bounded accompaniment. Selection
    uses velocity (a salience proxy), never a claim of calibrated confidence.
    """
    import pretty_midi

    step = 60 / tempo_bpm / (4 if grid == "sixteenth" else 2)
    minimum = 0.045 if role == "piano" or detail == "detailed" else 0.09
    velocity_floor = 12 if role == "piano" else 24 if detail == "detailed" else 32
    ranges = {"piano": (21, 108), "melody": (45, 96), "vocals": (45, 96),
              "bass": (21, 60), "other": (36, 96)}
    low, high = ranges[role]
    valid = []
    for item in notes:
        if (not math.isfinite(item.start) or not math.isfinite(item.end)
                or item.end - item.start < minimum or not low <= item.pitch <= high
                or item.velocity < velocity_floor):
            continue
        # The chosen score grid also defines MIDI timing. Group near-simultaneous
        # attacks before choosing a melody, so a few ms of jitter cannot become
        # a dense run of unrelated tiny notes.
        start = max(0, math.floor(item.start / step + 0.5)) * step
        end = max(start + step, math.floor(item.end / step + 0.5) * step)
        valid.append(pretty_midi.Note(velocity=int(item.velocity), pitch=int(item.pitch),
                                     start=start, end=end))
    grouped = defaultdict(list)
    for item in valid:
        grouped[round(item.start / step)].append(item)

    selected = []
    previous_pitch = None
    for _, attacks in sorted(grouped.items()):
        # Duplicate detections from different harmonics/overlapping windows do
        # not represent additional attacks of a piano key.
        by_pitch = {}
        for item in attacks:
            previous = by_pitch.get(item.pitch)
            if previous is None or item.velocity > previous.velocity:
                if previous is not None:
                    item.end = max(previous.end, item.end)
                by_pitch[item.pitch] = item
            elif previous is not None:
                previous.end = max(previous.end, item.end)
        attacks = list(by_pitch.values())
        if role in ("vocals", "melody", "bass"):
            def salience(item, previous_pitch=previous_pitch):
                continuity = min(abs(item.pitch - previous_pitch), 24) if previous_pitch is not None else 0
                # Continuity limits abrupt octave jumps caused by overtone
                # detections without forcing a vocal melody into a fixed octave.
                return item.velocity - 0.8 * continuity
            chosen = max(attacks, key=salience)
            selected.append(chosen)
            previous_pitch = chosen.pitch
        elif role == "other":
            limit = 5 if detail == "detailed" else 3
            selected.extend(sorted(attacks, key=lambda item: -item.velocity)[:limit])
        else:
            selected.extend(attacks)

    selected.sort(key=lambda item: (item.start, item.pitch))
    active = {}
    monophonic = role in ("vocals", "melody", "bass")
    for item in selected:
        # A repeated key replaces an older held key; a monophonic line replaces
        # the previous pitch. This removes tied clouds from sustained detections.
        key = 0 if monophonic else item.pitch
        previous = active.get(key)
        if previous is not None and previous.end > item.start:
            previous.end = item.start
        active[key] = item
    return [item for item in selected if item.end > item.start]


def estimate_grid_phase(notes: list, *, tempo_bpm: float, grid: str) -> float:
    """Find a shared sub-grid onset offset, without claiming to find a downbeat.

    Notes close to a quantizer boundary otherwise alternate between neighbouring
    slots as detection jitter changes sign. Correct only a strong common phase;
    rubato and irregular rhythms are left alone. Work is bounded to 2048 notes.
    """
    if len(notes) < 8:
        return 0.0
    indices = np.linspace(0, len(notes) - 1, min(len(notes), 2048), dtype=int)
    selected = [notes[index] for index in indices
                if math.isfinite(notes[index].start) and notes[index].velocity >= 24]
    if len({round(item.start, 3) for item in selected}) < 8:
        return 0.0
    step = 60 / tempo_bpm / (4 if grid == "sixteenth" else 2)
    phases = np.asarray([item.start % step for item in selected]) * (2 * np.pi / step)
    weights = np.asarray([item.velocity for item in selected], dtype=float)
    resultant = np.average(np.exp(1j * phases), weights=weights)
    if abs(resultant) < 0.6:
        return 0.0
    return float(np.angle(resultant) * step / (2 * np.pi))


def reduce_accompaniment(notes: list, *, detail: str) -> list:
    """Cap simultaneously sounding accompaniment, not only simultaneous attacks."""
    import pretty_midi

    limit = 5 if detail == "detailed" else 3
    active = []
    output = []
    for item in sorted(notes, key=lambda n: (n.start, -n.velocity)):
        active = [old for old in active if old.end > item.start]
        if len(active) >= limit:
            weakest = min(active, key=lambda n: n.velocity)
            if item.velocity <= weakest.velocity:
                continue
            weakest.end = item.start
            active.remove(weakest)
        new = pretty_midi.Note(velocity=item.velocity, pitch=item.pitch,
                              start=item.start, end=item.end)
        output.append(new)
        active.append(new)
    return [item for item in output if item.end > item.start]


def estimate_key(notes: list) -> str | None:
    """Pitch-duration profile estimate; this never changes detected pitches."""
    from music21 import analysis, note, stream

    histogram = np.zeros(12)
    for item in notes:
        histogram[item.pitch % 12] += min(4, item.end - item.start)
    if np.count_nonzero(histogram) < 4:
        return None
    profile = stream.Stream()
    scale = max(1.0, float(histogram.max()))
    for pitch, weight in enumerate(histogram):
        if weight:
            profile.append(note.Note(pitch + 60, quarterLength=float(weight / scale)))
    detected = analysis.discrete.KrumhanslSchmuckler().getSolution(profile)
    if detected.correlationCoefficient < 0.65:
        return None
    return f"{detected.tonic.name} {detected.mode}"
