"""Native decoding/routing parity on original and licensed fixtures, no scores."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services import bass_verifier as service
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from prepare_robust_training_stems import digest, preserve


def tuples(midi):
    return [(part_index, note.pitch, note.start, note.end, note.velocity)
            for part_index, part in enumerate(midi.instruments) for note in part.notes]


def verify(directory, run):
    output = run / 'native-routing.json'
    if output.exists():
        raise ValueError('Preserve native release evidence')
    witness = json.loads((run / 'runtime-parity.json').read_text())
    export = json.loads((run / 'portable/export.json').read_text())
    if (not witness['passes'] or witness['checkpoint_sha256'] != export['checkpoint_sha256']
            or witness['batch_selection_sha256'] != export['batch_selection_sha256']
            or witness['runtime_source_sha256'] != digest(Path(service.__file__))
            or witness['arrays'] != {path.name: expected for path, expected, _ in service.CHECKPOINTS}):
        raise ValueError('Changed or failed portable/runtime witness')
    work = directory / 'bass-native-release-v1'
    work.mkdir(exist_ok=True)
    cases = [directory / 'bass-held-regression-v1' / ('bass-held-v1-' + name) / 'decoded.wav'
             for name in ('held', 'repeated', 'octave')]
    cases.append(directory / 'slakh-bass-v1/slakh-bass-v1-Track00006-oracle-000/decoded.wav')
    rate = 22050
    samples = np.zeros(32 * rate, dtype=np.float32)
    for start, end, pitch in [(3., 30.5, 36), (30.6, 31.5, 43)]:
        first, stop = round(start * rate), round(end * rate)
        t = np.arange(stop - first) / rate
        hz = pretty_midi.note_number_to_hz(pitch)
        envelope = np.minimum(1., t / .008) * np.minimum(1., np.maximum(0., (end - start - t) / .025))
        samples[first:stop] += ((.2 * np.sin(2 * np.pi * hz * t) + .08 * np.sin(4 * np.pi * hz * t)) * envelope).astype(np.float32)
    sf.write(work / 'window-32.wav', samples, rate, subtype='FLOAT')
    sf.write(work / 'short.wav', samples[3 * rate:3 * rate + round(.05 * rate)], rate, subtype='FLOAT')
    cases.extend([work / 'window-32.wav', work / 'short.wav'])
    engine, verifier, rows = _GeneralEngine('balanced'), service.BassVerifier(), []
    for path in cases:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            acoustic, baseline, _ = engine.predict_function(str(path), model_or_model_path=engine.model,
                minimum_frequency=float(pretty_midi.note_number_to_hz(21)),
                maximum_frequency=float(pretty_midi.note_number_to_hz(60)),
                onset_threshold=.5, frame_threshold=.3, minimum_note_length=90.,
                multiple_pitch_bends=False, melodia_trick=False, midi_tempo=90.)
            notes = [note for part in baseline.instruments for note in part.notes]
            waveform, sample_rate = sf.read(path, dtype='float32')
            x = note_features(waveform, sample_rate, acoustic, notes, include_context=True)
            p, g = [model.probability(x[:, :model.feature_count]) for model in verifier.models]
            duration = len(waveform) / sample_rate
            eligible = np.asarray([n.start >= 2.5 and n.end <= duration - 2.5 and 21 <= n.pitch < 60 for n in notes], dtype=bool)
            keep = ~(eligible & (p < .1) & (g < .1))
            before = tuples(baseline)
            expected = [event for event, accepted in zip(before, keep, strict=True) if accepted]
            actual = engine.predict(path, 'bass', 90.)
        assert tuples(actual) == expected, str(path)
        rows.append({'case': path.name, 'directory': path.parent.name, 'duration': duration,
                     'baseline_candidates': len(before), 'removed': int((~keep).sum()), 'passes': True})
    result = {'passes': True, 'cases': rows, 'checkpoint_sha256': export['checkpoint_sha256'],
              'portable_runtime_report_sha256': digest(run / 'runtime-parity.json'),
              'routing_source_sha256': digest(Path(__file__).resolve().parents[1] / 'backend/app/services/piano_transcription.py'),
              'script_sha256': digest(Path(__file__)), 'bass_metadata': engine.bass_verification,
              'checks': 'Actual Basic Pitch bass bounds, sample/audio features, portable filtering, production routing, unchanged retained event tuples; 32-second context and subframe audio. No scores or user uploads.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run)
