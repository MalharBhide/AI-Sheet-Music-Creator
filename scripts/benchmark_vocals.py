"""Known-note vocal-like synthesis: a regression benchmark, not human singing.

PYTHONPATH=backend python scripts/benchmark_vocals.py /tmp/vocal-quality
Uses real Basic Pitch and pYIN on deterministic harmonic/vibrato fixtures.
The baseline is the same neural output before the vocal refinement stage.
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pretty_midi
import soundfile as sf

from app.services.audio_analysis import clean_notes
from app.services.piano_transcription import _GeneralEngine
from app.services.vocal_refinement import refine_vocals


def reference(directory: Path, name: str, bpm: float, transpose: int,
              harmonics: list[float], vibrato: float, noise: float):
    rate, beat = 22050, 60 / bpm
    samples = np.zeros(int((12 * beat + 1) * rate), dtype=np.float32)
    expected = []
    rng = np.random.default_rng(41)
    for i, pitch in enumerate([55, 57, 59, 59, 62, 60, 57, 55, 67, 64]):
        pitch += transpose
        start = beat / 2 + i * beat
        length = beat * (.8 if i % 3 == 0 else .65)
        if name == 'rearticulated':
            pitch, length = 60, beat
        t = np.arange(int(length * rate)) / rate
        frequency = librosa.midi_to_hz(pitch) * 2 ** (
            vibrato * np.sin(2 * np.pi * 5.4 * t) / 12)
        phase = 2 * np.pi * np.cumsum(frequency) / rate
        tone = sum(amplitude * np.sin(h * phase)
                   for h, amplitude in enumerate(harmonics, start=1))
        envelope = np.minimum(t / .025, 1) * np.minimum((length - t) / .025, 1)
        tone += rng.normal(0, noise, len(tone))
        begin = int(start * rate)
        samples[begin:begin + len(t)] += .2 * tone * envelope
        expected.append(pretty_midi.Note(88, pitch, start, start + length))
    path = directory / f'{name}.wav'
    sf.write(str(path), samples, rate)
    return path, expected


def metrics(expected: list, actual: list) -> dict:
    def intervals(notes):
        return np.array([(n.start, n.end) for n in notes]).reshape(-1, 2)

    def pitches(notes):
        return np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    result = {'expected_notes': len(expected), 'detected_notes': len(actual)}
    for name, ratio in [('pitch_onset', None), ('pitch_onset_offset', .2)]:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            intervals(expected), pitches(expected), intervals(actual), pitches(actual),
            onset_tolerance=.08, offset_ratio=ratio, offset_min_tolerance=.05)
        result[name] = dict(precision=round(precision, 4), recall=round(recall, 4),
                            f1=round(f1, 4))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    engine = _GeneralEngine('balanced')
    results = {}
    fixtures = [
        ('clear-low', 80, -3, [1, .3, .15, .08], .15, 0),
        ('clear-high', 120, 12, [1, .3, .15, .08], .25, 0),
        ('strong-overtone', 80, 0, [.12, .65, .3, .12], .3, 0),
        ('fast-overtone', 144, 5, [.12, .65, .3, .12], .3, 0),
        ('wide-vibrato', 96, 2, [.5, .6, .3, .12], .55, 0),
        ('breathy', 110, 7, [.7, .35, .15, .08], .3, .05),
        ('rearticulated', 120, 0, [1, .3, .15, .08], .2, 0),
    ]
    for name, bpm, transpose, harmonics, vibrato, noise in fixtures:
        path, expected = reference(args.directory, name, bpm, transpose, harmonics, vibrato, noise)
        started = time.monotonic()
        # Melody uses the same Basic Pitch settings, without vocal refinement.
        midi = engine.predict(path, 'melody', bpm)
        notes = [n for part in midi.instruments for n in part.notes]
        baseline = clean_notes(notes, role='vocals', detail='balanced', tempo_bpm=bpm,
                               grid='sixteenth')
        refine_vocals(path, midi)
        refined = clean_notes([n for p in midi.instruments for n in p.notes], role='vocals',
                              detail='balanced', tempo_bpm=bpm, grid='sixteenth')
        results[name] = {'baseline': metrics(expected, baseline),
                         'refined': metrics(expected, refined),
                         'runtime_seconds': round(time.monotonic() - started, 2)}
        print(json.dumps({name: results[name]}), flush=True)
    (args.directory / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
