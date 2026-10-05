"""Native decoding and post-hold evidence against sealed residual sklearn weights."""

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services import bass_residual as service
from app.services.bass_articulation import BassArticulation
from app.services.bass_verifier import BassVerifier
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from prepare_robust_training_stems import digest, preserve


def verify(directory, run, fresh):
    output = run / 'native-routing.json'
    if output.exists():
        raise ValueError('Preserve native residual witness')
    witness = json.loads((run / 'runtime-parity.json').read_text())
    export = json.loads((run / 'portable/export.json').read_text())
    batch = json.loads((run / 'batch-selection.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    winner = batch['winner']
    checkpoint = run / winner['name'] / 'candidate.pickle'
    arrays = {path.name: expected for path, expected, _, _ in service.CHECKPOINTS}
    if (not witness['passes'] or not export['passes'] or not batch['selected']
            or batch['test_used_for_selection'] or winner['test_used_for_selection']
            or digest(checkpoint) != winner['checkpoint_sha256']
            or witness['checkpoint_sha256'] != winner['checkpoint_sha256']
            or export['checkpoint_sha256'] != winner['checkpoint_sha256']
            or batch['plan_sha256'] != digest(run / 'plan.json')
            or witness['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or export['batch_selection_sha256'] != witness['batch_selection_sha256']
            or witness['runtime_source_sha256'] != digest(Path(service.__file__))
            or witness['arrays'] != arrays or export['sha256'] != arrays
            or (winner['threshold'], winner['guardian_threshold']) != (.05, .1)):
        raise ValueError('Changed or failed frozen residual witnesses')
    # The production route has now changed. All other frozen baseline sources
    # must still match; pre-routing parity was sealed before that change.
    root = Path(__file__).resolve().parents[1]
    for baseline in (plan['baseline_hashes'], plan['baseline_hashes']['v10']):
        for name, expected in baseline['sources'].items():
            if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
                raise ValueError('Changed frozen baseline source')
    for name, expected in plan['code_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError('Changed frozen fitting code')
    reports = {}
    for name in ('consumed-regression', 'original-first-pass'):
        report = json.loads((run / (name + '.json')).read_text())
        if (not report['passes'] or report['false_notes_removed'] <= 0
                or not report['per_recording'] or any(not r['passes'] for r in report['per_recording'])
                or report['checkpoint_sha256'] != winner['checkpoint_sha256']
                or report['batch_selection_sha256'] != witness['batch_selection_sha256']
                or (report['threshold'], report['guardian_threshold']) != (.05, .1)):
            raise ValueError('Changed or failed preservation evidence')
        reports[name] = report
    first = reports['original-first-pass']
    if not first['first_pass_complete'] or first['test_manifest_sha256'] != digest(fresh / 'manifest.json'):
        raise ValueError('Changed first-pass corpus')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    cases = [directory / 'bass-held-regression-v1' / ('bass-held-v1-' + name) / 'decoded.wav'
             for name in ('held', 'repeated', 'octave')]
    cases.append(directory / 'slakh-bass-v1/slakh-bass-v1-Track00006-oracle-000/decoded.wav')
    selected = sorted(r['id'] for r in first['per_recording']
                      if r['baseline_bass']['false_positives'] > r['candidate']['false_positives'])[:2]
    if len(selected) != 2:
        raise ValueError('Native routing must exercise actual learned residual rejections')
    cases.extend(fresh / name / 'decoded.wav' for name in selected)
    work = directory / 'bass-residual-native-v1'
    work.mkdir(exist_ok=True)
    rate = 22050
    samples = np.zeros(32 * rate, dtype=np.float32)
    for start, end, pitch in ((3., 30.5, 36), (30.6, 31.5, 43)):
        first_sample, stop = round(start * rate), round(end * rate)
        t = np.arange(stop - first_sample) / rate
        hz = pretty_midi.note_number_to_hz(pitch)
        envelope = np.minimum(1., t / .008) * np.minimum(1., np.maximum(0., (end - start - t) / .025))
        samples[first_sample:stop] += ((.2 * np.sin(2 * np.pi * hz * t) + .08 * np.sin(4 * np.pi * hz * t)) * envelope).astype(np.float32)
    sf.write(work / 'window-32.wav', samples, rate, subtype='FLOAT')
    sf.write(work / 'short.wav', samples[3 * rate:3 * rate + round(.05 * rate)], rate, subtype='FLOAT')
    cases.extend((work / 'window-32.wav', work / 'short.wav'))
    engine, baseline, articulation, rows = _GeneralEngine('balanced'), BassVerifier(), BassArticulation(), []
    for path in cases:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            acoustic, midi, _ = engine.predict_function(str(path), model_or_model_path=engine.model,
                minimum_frequency=float(pretty_midi.note_number_to_hz(21)),
                maximum_frequency=float(pretty_midi.note_number_to_hz(60)),
                onset_threshold=.5, frame_threshold=.3, minimum_note_length=90.,
                multiple_pitch_bends=False, melodia_trick=False, midi_tempo=90.)
            assert len(midi.instruments) <= 1
            raw_count = sum(len(part.notes) for part in midi.instruments)
            v10_rejected = baseline.filter(path, acoustic, midi)
            merged = articulation.filter(path, acoustic, midi, baseline)
            notes = [n for part in midi.instruments for n in part.notes]
            waveform, sample_rate = sf.read(path, dtype='float32')
            duration = len(waveform) / sample_rate
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
            x = note_features(waveform, sample_rate, acoustic, notes, include_context=True)
            p, g = [model.probability(x[:, :model.feature_count]) for model in baseline.models]
            x = service.features(events, x, p, g)
            confidence = [model.predict_proba(view)[:, 1] if len(x) else np.empty(0)
                          for model, view in zip(models, (x, service.acoustic_view(x)), strict=True)]
            eligible = (events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5) & (events[:, 2] >= 21) & (events[:, 2] < 60)
            keep = ~(eligible & (confidence[0] < .05) & (confidence[1] < .1))
            actual = engine.predict(path, 'bass', 90.)
        actual_events = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                                  for part in actual.instruments for n in part.notes], float).reshape(-1, 4)
        np.testing.assert_array_equal(actual_events, events[keep], err_msg=str(path))
        assert len(actual.instruments) <= 1
        rows.append({'case': path.name, 'directory': path.parent.name, 'duration': duration,
                     'raw_candidates': raw_count, 'v10_rejected': v10_rejected,
                     'merged_boundaries': merged, 'residual_rejected': int((~keep).sum()), 'passes': True})
    if not sum(row['residual_rejected'] for row in rows):
        raise ValueError('Native routing did not exercise learned residual filtering')
    result = {'passes': True, 'cases': rows, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'portable_runtime_report_sha256': digest(run / 'runtime-parity.json'),
              'routing_source_sha256': digest(root / 'backend/app/services/piano_transcription.py'),
              'runtime_source_sha256': digest(Path(service.__file__)), 'script_sha256': digest(Path(__file__)),
              'bass_metadata': engine.bass_verification,
              'checks': 'Native bounded Basic Pitch and actual post-hold audio/features; frozen sklearn residual decisions match the complete V10 + V11 + residual route. Exact retained pitches/attacks/ends/velocities. Original/licensed fixtures only; no user uploads or scores.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run, args.fresh)
