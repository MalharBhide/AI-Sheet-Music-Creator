"""Conservative discrete attack correction, fitted only on licensed train clips."""

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
from app.services.melody_decoder import FEATURE_VERSION
from app.services.vocal_melody import CHECKPOINT
from pitch_interval_coverage import compare
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from train_melody_timing import matches

SHIFTS = np.array([-.02, 0., .02])
THRESHOLDS = (.8, .9, .95, .975, .99)
SCALES = (.5, 1.)


def correct(events, choices, scale=1.):
    """Move an attack and its touching previous release together, within 20 ms."""
    result = events.copy()
    for index, shift in enumerate(np.asarray(choices) * scale):
        if not shift:
            continue
        start = events[index, 0] + shift
        if start < 0 or result[index, 1] - start < .02:
            continue
        if index and abs(events[index - 1, 1] - events[index, 0]) <= 1e-6:
            if start - result[index - 1, 0] < .02:
                continue
            result[index - 1, 1] = start
        elif index and result[index - 1, 1] > start:
            continue
        result[index, 0] = start
    if (np.any(result[:, 1] <= result[:, 0])
            or np.any(result[1:, 0] < result[:-1, 1] - 1e-6)
            or np.any(np.abs(result[:, :2] - events[:, :2]) > .020001)):
        raise ValueError('Invalid bounded attack correction')
    return result


def protected(reference, before, after):
    preserved = True
    counts = {}
    for name, offsets in (('onset', False), ('hold', True)):
        old = {i for i, _ in matches(reference, before, offsets=offsets)}
        new = {i for i, _ in matches(reference, after, offsets=offsets)}
        counts[name] = {'before': len(old), 'after': len(new), 'lost': len(old-new)}
        preserved &= old.issubset(new)
    coverage = compare(reference, before, after)
    return bool(preserved and coverage['passes']), counts, coverage


def data(directory, group):
    items = []
    manifests = [json.loads((directory / name).read_text()) for name in ('melody-v1/split.json', 'vocalset-split.json')]
    if manifests[1]['annotation_clock_version'] != 1:
        raise ValueError('Uncorrected annotation clocks')
    for corpus, manifest in zip(('vocadito', 'vocalset'), manifests, strict=True):
        for entry in manifest['tracks'][group]:
            identity = entry if corpus == 'vocadito' else entry['id']
            name = corpus + '-' + str(identity)
            path = directory / 'melody-timing-v1/cached-boundaries' / (name + '.npz')
            feature = directory / ('features/vocadito_' + str(identity) + '.npz' if corpus == 'vocadito'
                                   else 'vocalset-features/' + str(identity) + '.npz')
            with np.load(feature, allow_pickle=False) as saved:
                if str(saved['version']) != FEATURE_VERSION:
                    raise ValueError('Changed source feature version')
                annotations = saved['A1' if corpus == 'vocadito' else 'reference']
            reference = np.array([[s, s+d, float(librosa.hz_to_midi(hz))] for s, hz, d in annotations]).reshape(-1, 3)
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['model_sha256']) != digest(CHECKPOINT):
                    raise ValueError('Changed melody timing baseline')
                item = {key: saved[key] for key in ('events', 'reference', 'onset_x', 'duration')}
            np.testing.assert_array_equal(item['reference'], reference, err_msg=name)
            if (item['onset_x'].shape != (len(item['events']), 131)
                    or not np.isfinite(item['onset_x']).all() or not np.isfinite(item['events']).all()):
                raise ValueError('Invalid timing feature cache')
            items.append({**item, 'id': name, 'corpus': corpus, 'cache_sha256': digest(path),
                          'source_feature_sha256': digest(feature)})
    return items


def targets(item):
    events = item['events']
    y = np.ones(len(events), dtype=int)
    for ref, event in matches(item['reference'], events, .2):
        delta = item['reference'][ref, 0] - events[event, 0]
        if abs(delta) < .03:
            continue
        proposal = np.zeros(len(events))
        proposal[event] = np.sign(delta) * .02
        changed = correct(events, proposal)
        if changed[event, 0] == events[event, 0]:
            continue
        if protected(item['reference'], events, changed)[0]:
            y[event] = 0 if delta < 0 else 2
    return y


def evaluate(items, model, threshold, scale):
    rows = []
    for item in items:
        old = item['events']
        p = model.predict_proba(item['onset_x']) if len(old) else np.empty((0, 3))
        labels = model.classes_[p.argmax(axis=1)] if len(old) else np.empty(0, int)
        shifts = np.where(p.max(axis=1) >= threshold, SHIFTS[labels], 0.) if len(old) else np.empty(0)
        new = correct(old, shifts, scale)
        passes, counts, coverage = protected(item['reference'], old, new)
        pairs = matches(item['reference'], old, .2)
        before = sum(abs(old[j, 0]-item['reference'][i, 0]) for i, j in pairs)
        after = sum(abs(new[j, 0]-item['reference'][i, 0]) for i, j in pairs)
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'passes': passes, 'matches': counts,
                     'coverage': coverage, 'timing_pairs': len(pairs), 'before_error_sum': before,
                     'after_error_sum': after, 'changed_attacks': int(np.sum(new[:, 0] != old[:, 0]))})
    totals = {}
    for corpus in sorted({r['corpus'] for r in rows}):
        group = [r for r in rows if r['corpus'] == corpus]
        n = sum(r['timing_pairs'] for r in group)
        totals[corpus] = {'timing_pairs': n, 'before_mae': sum(r['before_error_sum'] for r in group)/max(1, n),
                         'after_mae': sum(r['after_error_sum'] for r in group)/max(1, n)}
    return {'threshold': threshold, 'scale': scale, 'passes': all(r['passes'] for r in rows),
            'changed_attacks': sum(r['changed_attacks'] for r in rows), 'aggregate': totals,
            'per_recording': rows}


def train(directory, output):
    output.mkdir(exist_ok=False)
    training, validation = [data(directory, g) for g in ('train', 'validation')]
    settings = {'max_iter': 200, 'max_leaf_nodes': 7, 'min_samples_leaf': 40, 'l2_regularization': 8.,
                'learning_rate': .04, 'early_stopping': False, 'random_state': 261005}
    root = Path(__file__).resolve().parents[1]
    source_names = ('training/train_attack_timing_v2.py', 'training/train_melody_timing.py',
                    'training/pitch_interval_coverage.py', 'backend/app/services/melody_decoder.py')
    plan = {'settings': settings, 'shifts_seconds': SHIFTS.tolist(), 'thresholds': list(THRESHOLDS),
            'scales': list(SCALES), 'baseline_sha256': digest(CHECKPOINT),
            'source_sha256': {name: digest(root / name) for name in source_names},
            'split_sha256': {name: digest(directory / name) for name in ('melody-v1/split.json', 'vocalset-split.json')},
            'inputs': [{k: item[k] for k in ('id', 'corpus', 'cache_sha256', 'source_feature_sha256')}
                       for item in training+validation], 'train_clips': len(training), 'validation_clips': len(validation),
            'rights': 'Existing CC-BY-4.0 Vocadito and VocalSet sources; source-group partitions and corrected clocks retained.',
            'selection': 'Validation only. No lost matched attacks, holds or per-reference pitch coverage in any recording; no corpus onset MAE regression; positive gain required.',
            'test_used': False, 'no_user_audio_or_scores': True,
            'scope': 'Three-class attack correction. Fixed melody pitches and event counts; adjacent touching release moves with attack; at most 20 ms. Training and validation only; not deployed.'}
    preserve(output / 'plan.json', plan)
    xs, ys, weights = [], [], []
    for item in training:
        if not len(item['events']):
            continue
        xs.append(item['onset_x'])
        ys.append(targets(item))
        corpus_n = sum(i['corpus'] == item['corpus'] for i in training)
        weights.append(np.full(len(item['events']), 1/(corpus_n*len(item['events']))))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=w/w.mean())
    with (output / 'candidate.pickle').open('wb') as stream:
        pickle.dump(model, stream)
    reports = [evaluate(validation, model, threshold, scale) for threshold in THRESHOLDS for scale in SCALES]
    preserve(output / 'validation.json', reports)
    eligible = [r for r in reports if r['passes'] and r['changed_attacks'] > 0
                and all(v['after_mae'] <= v['before_mae'] + 1e-12 for v in r['aggregate'].values())
                and sum(v['before_mae']-v['after_mae'] for v in r['aggregate'].values()) > 1e-9]
    winner = min(eligible, key=lambda r: sum(v['after_mae'] for v in r['aggregate'].values())) if eligible else None
    result = {'selected': winner is not None, 'winner': {k:v for k,v in winner.items() if k != 'per_recording'} if winner else None,
              'training_events': len(y), 'training_class_counts': np.bincount(y, minlength=3).tolist(),
              'checkpoint_sha256': digest(output / 'candidate.pickle'), 'plan_sha256': digest(output / 'plan.json'),
              'test_used': False, 'not_deployed': True}
    preserve(output / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    train(args.directory, args.output)
