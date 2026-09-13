"""Check neural vocal events against an independent monophonic pitch track."""

from pathlib import Path

import numpy as np
import soundfile as sf


def refine_notes(notes: list, pitches: np.ndarray, confidence: np.ndarray,
                 hop_seconds: float, rms: np.ndarray | None = None) -> list:
    """Remove clearly contradicted harmonics; trim unsupported outer tails.

    Uncertain tracking keeps the neural event. No pitch is invented or shifted,
    and this must only be used on an isolated, predominantly monophonic vocal.
    """
    import pretty_midi

    output = []
    for note in notes:
        first = max(0, int(np.ceil(note.start / hop_seconds)))
        stop = min(len(pitches), int(np.ceil(note.end / hop_seconds)))
        if stop <= first:
            output.append(note)
            continue
        tracked = pitches[first:stop]
        reliable = np.isfinite(tracked) & (confidence[first:stop] >= .2)
        matching = reliable & (np.abs(tracked - note.pitch) <= .75)
        # Require substantial evidence for a different fundamental before
        # deleting anything. Breath/noise and uncertain tracking are not proof.
        if reliable.mean() >= .6 and matching.sum() < reliable.sum() * .2:
            continue
        supported = np.flatnonzero(matching)
        if len(supported) * hop_seconds < .06:
            output.append(note)
            continue
        start = max(note.start, (first + supported[0] - 2) * hop_seconds)
        end = min(note.end, (first + supported[-1] + 2) * hop_seconds)
        if end - start >= .06:
            output.append(pretty_midi.Note(velocity=int(note.velocity), pitch=int(note.pitch),
                                          start=float(start), end=float(end)))
    if rms is None:
        return output
    merged = []
    for note in sorted(output, key=lambda n: (n.start, n.pitch)):
        previous = merged[-1] if merged else None
        if previous is not None and previous.pitch == note.pitch and 0 <= note.start - previous.end <= .05:
            boundary = round(note.start / hop_seconds)
            first, stop = max(0, boundary - 4), min(len(pitches), boundary + 4)
            steady = ((np.abs(pitches[first:stop] - note.pitch) <= .75)
                      & (confidence[first:stop] >= .2))
            before = rms[max(0, boundary - 5):boundary]
            after = rms[boundary:min(len(rms), boundary + 5)]
            # Vibrato can trigger repeated neural attacks on the same key.
            # Join only a continuously voiced pitch with no new amplitude
            # attack. A breath, pitch change or renewed attack keeps the split.
            attack = (not len(before) or not len(after)
                      or after.max() > max(1e-7, float(before.min())) * 1.4)
            if len(steady) and steady.mean() >= .9 and not attack:
                previous.end = max(previous.end, note.end)
                continue
        merged.append(note)
    return merged


def refine_vocals(path: Path, midi) -> None:
    """Run within the existing bounded audio window; no extra model download."""
    import librosa

    samples, rate = sf.read(str(path), dtype='float32')
    if samples.ndim != 1 or rate != 22050:
        raise ValueError('Vocal refinement requires a normalized mono stem')
    hop = 256
    f0, voiced, probability = librosa.pyin(
        samples, sr=rate, fmin=float(librosa.midi_to_hz(45)),
        fmax=float(librosa.midi_to_hz(96)), frame_length=2048,
        hop_length=hop, resolution=.2)
    pitches = librosa.hz_to_midi(f0)
    confidence = np.where(voiced, probability, 0)
    rms = librosa.feature.rms(y=samples, frame_length=512, hop_length=hop)[0]
    for part in midi.instruments:
        part.notes = refine_notes(part.notes, pitches, confidence, hop / rate, rms)
