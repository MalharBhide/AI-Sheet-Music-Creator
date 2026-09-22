"""Bounded audio analysis and musical cleanup shared by transcription engines."""

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def _tempo_from_onsets(envelope: np.ndarray, sample_rate: int) -> float | None:
    """Compare beat hypotheses against the audio instead of one 120 BPM prior."""
    import librosa
    from scipy.stats import theilslopes

    hypotheses = []
    for prior in (60, 90, 120, 180):
        tempo, beats = librosa.beat.beat_track(onset_envelope=envelope,
                                             sr=sample_rate, hop_length=256,
                                             trim=True, start_bpm=prior)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        if len(beats) < 4 or not 30 <= bpm <= 240:
            continue
        spacing = float(theilslopes(beats)[0])
        fitted = 60 * sample_rate / (256 * spacing) if spacing > 0 else bpm
        if abs(fitted - bpm) < 0.08 * bpm:
            bpm = fitted
        if not 30 <= bpm <= 240:
            continue
        # Strong attacks support a beat. Reward covering the recording's attacks
        # as well, so merely choosing every second strong beat does not win.
        strengths = np.array([np.max(envelope[max(0, b - 2):b + 3]) for b in beats])
        regularity = np.median(np.abs(np.diff(beats) - np.median(np.diff(beats))))
        regularity /= max(1.0, float(np.median(np.diff(beats))))
        support = float(strengths.mean() * np.sqrt(strengths.sum() / envelope.sum()))
        support *= max(0.0, 1 - 4 * regularity)
        hypotheses.append((support, bpm))
    if not hypotheses:
        return None
    return max(hypotheses, key=lambda item: (item[0], -abs(item[1] - 120)))[1]


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
            bpm = _tempo_from_onsets(envelope, audio.samplerate)
            if bpm is not None:
                candidates.append(bpm)
    if not candidates:
        return 120.0, ["A steady tempo could not be detected; 120 BPM was used. Set the tempo manually for a better rhythmic result."]
    # Select an observed tempo instead of averaging incompatible beat levels.
    # For example, 90 and 180 BPM are an ambiguity; their median135 is neither.
    anchor = min(candidates, key=lambda candidate: (
        sum(abs(math.log(candidate / other)) for other in candidates), abs(candidate - 120)))
    # Sections can latch onto alternate beats or twice the beat rate. Align
    # only clear octave ambiguities before consensus; never average 90 and 180.
    aligned = []
    for candidate in candidates:
        nearest = min((candidate / 2, candidate, candidate * 2),
                      key=lambda value: abs(math.log(value / anchor)))
        aligned.append(nearest if 30 <= nearest <= 240 and abs(nearest / anchor - 1) < .08
                       else candidate)
    bpm = min(aligned, key=lambda candidate: sum(abs(math.log(candidate / other))
                                               for other in aligned))
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
    # The supervised vocal decoder already rejects brief pitch glitches. A
    # second generic 90 ms filter deletes legitimate fast melody notes.
    minimum = 0.045 if role in ("piano", "vocals") or detail == "detailed" else 0.09
    velocity_floor = 12 if role == "piano" else 24 if detail == "detailed" else 32
    ranges = {"piano": (21, 108), "melody": (45, 96), "vocals": (21, 108),
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


def reduce_accompaniment(notes: list, *, detail: str, melody: list | None = None) -> list:
    """Choose complete supporting lines without stealing a held note's release.

    Each pass selects a non-overlapping line by weighted interval scheduling.
    Duration times velocity favors sustained support over tiny loud fragments.
    Balanced arrangements keep two supporting voices below an available melody;
    the separate bass and melody are never reduced here. This is arrangement,
    not a new claim about which detected pitches are correct.
    """
    import pretty_midi

    limit = 5 if detail == 'detailed' else 2 if melody else 3
    candidates = sorted(notes, key=lambda n: (n.end, n.start, n.pitch))
    if detail == 'balanced' and melody:
        # clean_notes has already made this line monophonic, so endpoints are
        # sorted too. Consider the entire overlap, not just the support attack.
        lead = sorted(melody, key=lambda n: n.start)
        starts, ends = [n.start for n in lead], [n.end for n in lead]
        candidates = [item for item in candidates
                      if all(item.pitch < n.pitch for n in lead[
                          bisect_right(ends, item.start):bisect_left(starts, item.end)])]
    # Do not thin a passage that already fits: repeated maximum-weight line
    # selection is a reduction heuristic, not an optimal multi-voice partition.
    active, peak = 0, 0
    for _, change in sorted((at, change) for item in candidates
                             for at, change in ((item.start, 1), (item.end, -1))):
        active += change
        peak = max(peak, active)
    selected = candidates if peak <= limit else []
    remaining = [] if peak <= limit else candidates
    for _ in range(limit):
        candidates = remaining
        if not candidates:
            break
        ends = [n.end for n in candidates]
        scores = [0.]
        predecessors, take = [], []
        for index, item in enumerate(candidates):
            previous = bisect_right(ends, item.start, hi=index)
            score = scores[previous] + (item.end - item.start) * item.velocity
            predecessors.append(previous)
            take.append(score > scores[-1])
            scores.append(max(scores[-1], score))
        chosen, index = set(), len(candidates)
        while index:
            if take[index - 1]:
                chosen.add(index - 1)
                index = predecessors[index - 1]
            else:
                index -= 1
        selected.extend(candidates[i] for i in sorted(chosen))
        remaining = [item for i, item in enumerate(candidates) if i not in chosen]
    return [pretty_midi.Note(velocity=item.velocity, pitch=item.pitch,
                             start=item.start, end=item.end)
            for item in sorted(selected, key=lambda n: (n.start, n.pitch))]


def balance_piano_arrangement(parts: dict) -> None:
    """Give the lead a foreground dynamic while keeping expressive variation.

    Separate model passes do not produce comparable instrument loudness. Center
    each role on a piano dynamic, retaining local velocity differences within a
    modest range. This is an arrangement choice, not a confidence calibration.
    Timing, pitch and genuine held notes are left intact.
    """
    for role, target in {"vocals": 92, "bass": 68, "other": 54}.items():
        part = parts.get(role)
        if part is None or not part.notes:
            continue
        center = float(np.median([item.velocity for item in part.notes]))
        for item in part.notes:
            deviation = max(-14, min(14, (item.velocity - center) * 0.6))
            item.velocity = round(target + deviation)


def arrange_melody_register(parts: dict) -> int:
    """Place a low sung melody in the piano's middle register, consistently.

    One whole-line octave transposition preserves intervals and phrasing. Never
    fold individual notes into an octave or alter a solo piano transcription.
    The caller records this arrangement choice separately from recognition.
    """
    lead = parts.get('vocals')
    if lead is None or not lead.notes:
        return 0
    pitches = np.asarray([item.pitch for item in lead.notes])
    durations = np.asarray([item.end - item.start for item in lead.notes])
    order = np.argsort(pitches)
    median = pitches[order[np.searchsorted(np.cumsum(durations[order]), durations.sum() / 2)]]
    shift = max(0, math.ceil((60 - int(median)) / 12)) * 12
    # Preserve high excursions and stay in a comfortable piano register.
    shift = min(shift, max(0, (96 - int(pitches.max())) // 12) * 12)
    for item in lead.notes:
        item.pitch += shift
    return shift


def preserve_melody_releases(parts: dict) -> int:
    """Give melody attacks/releases priority when support uses the same key.

    A piano has one damper per key: a long supporting note can otherwise hold a
    melody key past its intended release. End earlier support at the melody
    attack and omit support attacks inside that note. Never invent a new attack
    at the end of the melody or shorten the melody itself.
    """
    lead = parts.get('vocals')
    if lead is None or not lead.notes:
        return 0
    by_pitch = defaultdict(list)
    for item in sorted(lead.notes, key=lambda item: item.start):
        by_pitch[item.pitch].append(item)
    starts = {pitch: [item.start for item in notes] for pitch, notes in by_pitch.items()}
    changed = 0
    for role in ('bass', 'other'):
        part = parts.get(role)
        if part is None:
            continue
        kept = []
        for item in part.notes:
            notes = by_pitch.get(item.pitch, [])
            if notes:
                index = bisect_left(starts[item.pitch], item.start)
                inside = (index > 0 and notes[index - 1].end > item.start)
                simultaneous = index < len(notes) and notes[index].start == item.start
                if inside or simultaneous:
                    changed += 1
                    continue
                if index < len(notes) and notes[index].start < item.end:
                    item.end = notes[index].start
                    changed += 1
            kept.append(item)
        part.notes = kept
    return changed


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
