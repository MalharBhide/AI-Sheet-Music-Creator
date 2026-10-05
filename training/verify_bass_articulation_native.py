"""Native bounded decoding and full bass routing against frozen sklearn, no scores."""

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services import bass_articulation as service
from app.services.bass_verifier import BassVerifier
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from bass_boundary_evidence import boundaries, features, merge
from bass_predictions import probability
from prepare_robust_training_stems import digest, preserve
from verify_bass_native_release import tuples


def verify(directory, run):
    output = run / 'native-routing.json'
    if output.exists():
        raise ValueError('Preserve native bass articulation witness')
    witness = json.loads((run / 'runtime-parity.json').read_text())
    export = json.loads((run / 'portable/export.json').read_text())
    batch = json.loads((run / 'batch-selection.json').read_text())
    winner = batch['winner']
    checkpoint = run / winner['name'] / 'candidate.pickle'
    if (not witness['passes'] or not export['passes'] or digest(checkpoint) != winner['checkpoint_sha256']
            or witness['checkpoint_sha256'] != winner['checkpoint_sha256']
            or export['checkpoint_sha256'] != winner['checkpoint_sha256']
            or witness['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or export['batch_selection_sha256'] != witness['batch_selection_sha256']
            or witness['runtime_source_sha256'] != digest(Path(service.__file__))
            or witness['arrays'] != {path.name: expected for path, expected, _ in service.CHECKPOINTS}):
        raise ValueError('Changed or failed frozen portable/runtime witness')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    work = directory / 'bass-articulation-native-v1'
    work.mkdir(exist_ok=True)
    cases = [directory / 'bass-held-regression-v1' / ('bass-held-v1-' + name) / 'decoded.wav'
             for name in ('held', 'repeated', 'octave')]
    cases.append(directory / 'slakh-bass-v1/slakh-bass-v1-Track00006-oracle-000/decoded.wav')
    first = json.loads((run / 'original-first-pass.json').read_text())
    selected = sorted(row['id'] for row in first['per_recording'] if row['merged_boundaries'])[:2]
    if len(selected) != 2:
        raise ValueError('Native routing must exercise actual learned hold merges')
    cases.extend(directory / 'bass-articulation-stress-v2' / name / 'decoded.wav' for name in selected)
    rate = 22050
    samples = np.zeros(32 * rate, dtype=np.float32)
    for start, end, pitch in [(3., 30.5, 36), (30.6, 31.5, 43)]:
        first_sample, stop = round(start * rate), round(end * rate)
        t = np.arange(stop - first_sample) / rate
        hz = pretty_midi.note_number_to_hz(pitch)
        envelope = np.minimum(1., t / .008) * np.minimum(1., np.maximum(0., (end - start - t) / .025))
        samples[first_sample:stop] += ((.2 * np.sin(2 * np.pi * hz * t) + .08 * np.sin(4 * np.pi * hz * t)) * envelope).astype(np.float32)
    sf.write(work / 'window-32.wav', samples, rate, subtype='FLOAT')
    sf.write(work / 'short.wav', samples[3 * rate:3 * rate + round(.05 * rate)], rate, subtype='FLOAT')
    cases.extend([work / 'window-32.wav', work / 'short.wav'])
    engine, baseline, rows = _GeneralEngine('balanced'), BassVerifier(), []
    for path in cases:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            acoustic, midi, _ = engine.predict_function(str(path), model_or_model_path=engine.model,
                minimum_frequency=float(pretty_midi.note_number_to_hz(21)),
                maximum_frequency=float(pretty_midi.note_number_to_hz(60)),
                onset_threshold=.5, frame_threshold=.3, minimum_note_length=90.,
                multiple_pitch_bends=False, melodia_trick=False, midi_tempo=90.)
            assert len(midi.instruments) <= 1
            notes = [note for part in midi.instruments for note in part.notes]
            waveform, sample_rate = sf.read(path, dtype='float32')
            x = note_features(waveform, sample_rate, acoustic, notes, include_context=True)
            p, g = [model.probability(x[:, :model.feature_count]) for model in baseline.models]
            duration = len(waveform) / sample_rate
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], dtype=float).reshape(-1, 4)
            eligible = np.asarray([n.start >= 2.5 and n.end <= duration - 2.5 and 21 <= n.pitch < 60 for n in notes], bool)
            keep = ~(eligible & (p < .1) & (g < .1))
            item = {'events': events, 'x': x, 'baseline_p': p, 'baseline_g': g, 'keep': keep, 'shared': eligible}
            pairs = boundaries(item)
            confidence = [probability(model, features(item, pairs, acoustic=acoustic_view))
                          for model, acoustic_view in zip(models, (False, True), strict=True)]
            expected, merged = merge(item, pairs, *confidence, winner['threshold'], winner['guardian_threshold'])
            actual = engine.predict(path, 'bass', 90.)
        actual_events = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                                    for part in actual.instruments for n in part.notes], dtype=float).reshape(-1, 4)
        np.testing.assert_array_equal(actual_events, expected, err_msg=str(path))
        assert all(part_index == 0 for part_index, *_ in tuples(actual)), str(path)
        rows.append({'case': path.name, 'directory': path.parent.name, 'duration': duration,
                     'raw_candidates': len(events), 'v10_rejected': int((~keep).sum()),
                     'merged_boundaries': len(merged), 'passes': True})
    if not sum(row['merged_boundaries'] for row in rows):
        raise ValueError('Native routing did not exercise learned merging')
    result = {'passes': True, 'cases': rows, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'portable_runtime_report_sha256': digest(run / 'runtime-parity.json'),
              'routing_source_sha256': digest(Path(__file__).resolve().parents[1] / 'backend/app/services/piano_transcription.py'),
              'runtime_source_sha256': digest(Path(service.__file__)), 'script_sha256': digest(Path(__file__)),
              'bass_metadata': engine.bass_verification,
              'checks': 'Native bounded Basic Pitch, actual audio/features, frozen sklearn pair decisions versus complete production V10 filter + articulation routing; exact retained pitches, attacks, velocities and hold extensions. Original/licensed fixtures only; no user uploads or scores.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run)
