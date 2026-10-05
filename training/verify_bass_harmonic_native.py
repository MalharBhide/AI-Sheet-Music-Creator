"""Native decoding and post-hold evidence against sealed harmonic sklearn weights."""

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services import bass_harmonic as service
from app.services.bass_articulation import BassArticulation
from app.services.bass_residual import BassResidual
from app.services.bass_texture import BassTexture
from app.services.bass_verifier import BassVerifier
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from prepare_robust_training_stems import digest, preserve


def verify(directory, run, fresh):
    output = run / 'native-routing.json'
    if output.exists():
        raise ValueError('Preserve native harmonic witness')
    seal = json.loads((run / 'release-seal.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in seal['witnesses'].items():
        if digest(run / name) != expected:
            raise ValueError('Changed sealed harmonic release witness')
    for name, expected in seal['sources'].items():
        if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
            raise ValueError('Changed sealed fitting or runtime source')
    for name, expected in seal['arrays'].items():
        if digest(root / 'backend/app/assets' / name) != expected:
            raise ValueError('Changed sealed baseline or harmonic model array')
    if seal['fresh_manifest_sha256'] != digest(fresh / 'manifest.json'):
        raise ValueError('Changed sealed first-pass corpus')
    batch = json.loads((run / 'batch-selection.json').read_text())
    winner = batch['winner']
    checkpoint = run / winner['name'] / 'candidate.pickle'
    if digest(checkpoint) != winner['checkpoint_sha256']:
        raise ValueError('Changed sealed sklearn weights')
    first = json.loads((run / 'original-first-pass.json').read_text())
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    cases = [directory / 'bass-held-regression-v1' / ('bass-held-v1-' + name) / 'decoded.wav'
             for name in ('held', 'repeated', 'octave')]
    cases.append(directory / 'slakh-bass-v1/slakh-bass-v1-Track00006-oracle-000/decoded.wav')
    selected = sorted(r['id'] for r in first['per_recording']
                      if r['baseline_bass']['false_positives'] > r['candidate']['false_positives'])[:2]
    if len(selected) != 2:
        raise ValueError('Native routing must exercise actual learned harmonic rejections')
    cases.extend(fresh / name / 'decoded.wav' for name in selected)
    work = directory / 'bass-harmonic-native-v1'
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
    engine, baseline, articulation, residual, texture, rows = _GeneralEngine('balanced'), BassVerifier(), BassArticulation(), BassResidual(), BassTexture(), []
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
            v12_rejected = residual.filter(path, acoustic, midi, baseline)
            v13_rejected = texture.filter(path, acoustic, midi)
            notes = [n for part in midi.instruments for n in part.notes]
            waveform, sample_rate = sf.read(path, dtype='float32')
            duration = len(waveform) / sample_rate
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
            x = note_features(waveform, sample_rate, acoustic, notes, include_context=True)
            x = service.features(events, x)
            confidence = [model.predict_proba(view)[:, 1] if len(x) else np.empty(0)
                          for model, view in zip(models, (x, service.acoustic_view(x)), strict=True)]
            eligible = (events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5) & (events[:, 2] >= 21) & (events[:, 2] < 60)
            keep = ~(eligible & (confidence[0] < .025) & (confidence[1] < .3))
            actual = engine.predict(path, 'bass', 90.)
        actual_events = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                                  for part in actual.instruments for n in part.notes], float).reshape(-1, 4)
        np.testing.assert_array_equal(actual_events, events[keep], err_msg=str(path))
        assert len(actual.instruments) <= 1
        rows.append({'case': path.name, 'directory': path.parent.name, 'duration': duration,
                     'raw_candidates': raw_count, 'v10_rejected': v10_rejected,
                     'merged_boundaries': merged, 'v12_rejected': v12_rejected, 'v13_rejected': v13_rejected, 'harmonic_rejected': int((~keep).sum()), 'passes': True})
    if not sum(row['harmonic_rejected'] for row in rows):
        raise ValueError('Native routing did not exercise learned harmonic filtering')
    result = {'passes': True, 'cases': rows, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'portable_runtime_report_sha256': digest(run / 'runtime-parity.json'),
              'routing_source_sha256': digest(root / 'backend/app/services/piano_transcription.py'),
              'runtime_source_sha256': digest(Path(service.__file__)), 'script_sha256': digest(Path(__file__)),
              'bass_metadata': engine.bass_verification, 'release_seal_sha256': digest(run / 'release-seal.json'),
              'checks': 'Native bounded Basic Pitch and actual post-hold audio/features; frozen sklearn harmonic decisions match the complete V10 + V11 + V12 + V13 + harmonic route. Exact retained pitches/attacks/ends/velocities. Original/licensed fixtures only; no user uploads or scores.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


def seal_release(run, fresh):
    from bass_harmonic_v13_release import load_frozen
    from export_bass_harmonic import fresh_items
    from fresh_bass_harmonic_v13 import require_regression

    plan, winner, _ = load_frozen(run)
    require_regression(run, winner)
    fresh_items(run, fresh, winner)
    labels = json.loads((run / 'label-parity.json').read_text())
    if not labels['passes'] or labels['recordings'] != 865 or labels['v13_retained_events'] != 47798:
        raise ValueError('Failed label protection parity')
    parity = json.loads((run / 'runtime-parity.json').read_text())
    export = json.loads((run / 'portable/export.json').read_text())
    arrays = {path.name: expected for path, expected, _, _ in service.CHECKPOINTS}
    if (not parity['passes'] or not export['passes'] or parity['recordings'] != 897
            or parity['checkpoint_sha256'] != winner['checkpoint_sha256']
            or parity['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or parity['arrays'] != arrays or export['sha256'] != arrays
            or parity['runtime_source_sha256'] != digest(Path(service.__file__))):
        raise ValueError('Failed pre-routing harmonic parity')
    names = ['plan.json', 'batch-selection.json', 'consumed-regression.json',
             'original-first-pass.json', 'label-parity.json', 'runtime-parity.json', 'portable/export.json',
             winner['name'] + '/validation.json', winner['name'] + '/selection.json']
    sources = {**plan['baseline_hashes']['sources'], **plan['code_sha256'],
               'backend/app/services/bass_harmonic.py': digest(Path(service.__file__))}
    result = {'passes': True, 'witnesses': {name: digest(run / name) for name in names},
              'sources': sources, 'arrays': {**plan['baseline_hashes']['arrays'], **arrays},
              'fresh_manifest_sha256': digest(fresh / 'manifest.json'),
              'scope': 'Sealed with valid V13 guards before production routing changes. Only routing source may change for native parity.'}
    preserve(run / 'release-seal.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.seal:
        seal_release(args.run, args.fresh)
    else:
        verify(args.directory, args.run, args.fresh)
