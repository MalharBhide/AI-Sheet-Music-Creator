"""Reproducible known-note soundfont check, not a general accuracy benchmark.

Run with the backend and its transcription extra installed, FFmpeg and MuseScore:
  PYTHONPATH=backend python scripts/benchmark_transcription.py /tmp/piano-quality \
      --checkpoint /opt/models/piano.pth --modes piano full_mix --baseline

The original 96 BPM fixture tests polyphony and onset timing. It is deliberately
small enough to run on CPU. The full-mix check adds original synthetic percussion;
commercial recordings, real singers and studio mixtures need separate evaluation.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from app.config import Settings
from app.models import ScoreOptions
from app.services.audio_preprocess import normalize_audio
from app.services.midi_to_score import midi_to_musicxml
from app.services.piano_transcription import transcribe


def reference(directory: Path, settings: Settings):
    midi = pretty_midi.PrettyMIDI(initial_tempo=96, resolution=480)
    piano = pretty_midi.Instrument(0, name='Piano')
    for index, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 76, 74, 72, 71, 69, 67, 64, 60]):
        start = index * .625
        piano.notes.append(pretty_midi.Note(88, pitch, start, start + .46875))
    for index, pitch in enumerate([36, 41, 43, 36]):
        start = index * 2.5
        piano.notes.append(pretty_midi.Note(78, pitch, start, start + 2.1875))
    midi.instruments.append(piano)
    midi_path = directory / 'reference.mid'
    midi.write(str(midi_path))
    xml_path, wav_path = directory / 'reference.musicxml', directory / 'reference.wav'
    midi_to_musicxml(midi_path, xml_path, ScoreOptions(tempo_bpm=96), 'Original quality fixture')
    result = subprocess.run([settings.renderer(), '-o', str(wav_path), str(xml_path)],
                            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors='replace')[-4000:])
    return midi, wav_path


def metrics(reference_midi, predicted_path: Path, elapsed: float) -> dict:
    predicted = pretty_midi.PrettyMIDI(str(predicted_path))
    expected = [n for part in reference_midi.instruments for n in part.notes]
    actual = [n for part in predicted.instruments for n in part.notes]
    available = set(range(len(actual)))
    matches, errors = 0, []
    for item in expected:
        candidates = [index for index in available if actual[index].pitch == item.pitch
                      and abs(actual[index].start - item.start) <= .12]
        if candidates:
            match = min(candidates, key=lambda index: abs(actual[index].start - item.start))
            available.remove(match)
            matches += 1
            errors.append(abs(actual[match].start - item.start))
    precision = matches / len(actual) if actual else 0
    recall = matches / len(expected)
    return {'expected_notes': len(expected), 'detected_notes': len(actual), 'matched_notes': matches,
            'pitch_onset_precision': round(precision, 4), 'pitch_onset_recall': round(recall, 4),
            'pitch_onset_f1': round(2 * precision * recall / (precision + recall), 4) if matches else 0,
            'mean_onset_error_seconds': round(float(np.mean(errors)), 4) if errors else None,
            'runtime_seconds': round(elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['piano'], choices=['piano', 'full_mix'])
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=None)
    expected, original = reference(args.directory, settings)
    mono = args.directory / 'normalized.wav'
    normalize_audio(original, mono, settings)
    results = {}
    for mode in args.modes:
        source = mono
        if mode == 'full_mix':
            samples, rate = sf.read(str(original), always_2d=True)
            rng = np.random.default_rng(12)
            for at in np.arange(0, 10, .625):
                begin = int(at * rate)
                count = min(int(.06 * rate), len(samples) - begin)
                click = rng.normal(0, .045, count) * np.exp(-np.arange(count) / (.009 * rate))
                samples[begin:begin + count] += click[:, None]
            mixed = args.directory / 'piano-and-percussion.wav'
            sf.write(str(mixed), samples, rate)
            source = args.directory / 'mix-normalized.wav'
            normalize_audio(mixed, source, settings, preserve_stereo=True)
        output = args.directory / f'{mode}.mid'
        started = time.monotonic()
        report = transcribe(source, output, ScoreOptions(transcription_mode=mode),
                            piano_model_path=args.checkpoint)
        results[mode] = {**metrics(expected, output, time.monotonic() - started), 'analysis': report}
        print(json.dumps({mode: results[mode]}, indent=2), flush=True)
    if args.baseline:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        started = time.monotonic()
        _, midi, _ = predict(str(mono), model_or_model_path=ICASSP_2022_MODEL_PATH,
                             minimum_frequency=27.5, maximum_frequency=4186.01,
                             minimum_note_length=100.0, multiple_pitch_bends=False, midi_tempo=120)
        output = args.directory / 'previous-basic-pitch.mid'
        midi.write(str(output))
        results['previous_basic_pitch'] = metrics(expected, output, time.monotonic() - started)
        print(json.dumps({'previous_basic_pitch': results['previous_basic_pitch']}, indent=2), flush=True)
    (args.directory / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
