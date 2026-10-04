"""Predeclared offline batch for V7-retained shared notes at all confidences.

The website policy is unchanged. Select one winner on validation, freeze it,
and only then consume regression. Failed checkpoints never replace V7.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
import torch
from audit_verifier_labels import audit
from cache_note_verifier import targets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_left_consensus_v8 import (
    GUARDIAN_THRESHOLDS,
    acoustic_view,
    hashes,
    prepare,
    probabilities,
    regression_items,
    score,
)
from train_note_verifier import load_data, write
from train_residual_verifier import THRESHOLDS, cached_external

POLICY = 'v7-shared-interior-all-confidence-offline-v1'
PROFILES = (
    {'name': 'regularized', 'positive_weight': 10., 'max_iter': 500,
     'max_leaf_nodes': 31, 'min_samples_leaf': 80, 'l2_regularization': 8.},
    {'name': 'capacity', 'positive_weight': 8., 'max_iter': 600,
     'max_leaf_nodes': 63, 'min_samples_leaf': 40, 'l2_regularization': 4.},
    {'name': 'recall', 'positive_weight': 16., 'max_iter': 450,
     'max_leaf_nodes': 31, 'min_samples_leaf': 60, 'l2_regularization': 8.},
    {'name': 'shallow', 'positive_weight': 10., 'max_iter': 800,
     'max_leaf_nodes': 15, 'min_samples_leaf': 30, 'l2_regularization': 4.},
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(directory, extras):
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    identities = {item['id'] for item in training + validation}
    if len(identities) != len(training) + len(validation):
        raise ValueError('Duplicate fitting/validation clips')
    for path in extras:
        for item in cached_external(directory, path, directory):
            if item['group'] not in ('train', 'validation') or item['id'] in identities:
                raise ValueError('Test data or duplicate passages cannot enter fitting/selection')
            identities.add(item['id'])
            item['y'], item['mask'] = targets(item['reference'], item['events'])
            (training if item['group'] == 'train' else validation).append(item)
    return training, validation


def matrix(items, positive_weight):
    features, labels, weights, counts = [], [], [], []
    for corpus in sorted({item['corpus'] for item in items}):
        records = [item for item in items if item['corpus'] == corpus]
        for item in records:
            mask = item['mask'] & item['keep'] & item['shared']
            x, y = item['x'][mask], item['y'][mask]
            if not len(y):
                continue
            features.append(x)
            labels.append(y)
            weights.append(np.where(y == 1, positive_weight, 1.) / (len(records) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus,
                           'positive': int(y.sum()), 'negative': int(len(y) - y.sum())})
    if not features:
        raise ValueError('No supervised eligible events')
    x, y, weight = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    return x, y, weight / weight.mean(), counts


def coverage(items):
    """Validation diagnosis only, separating confidence from other protections."""
    result = {key: {'matched': 0, 'unmatched': 0} for key in
              ('uncertain_shared_interior', 'confident_shared_interior', 'unshared_or_edge')}
    for item in items:
        keep = item['keep']
        labels, _ = targets(item['reference'], item['events'][keep])
        shared, confidence = item['shared'][keep], item['p'][keep]
        for name, mask in [('uncertain_shared_interior', shared & (confidence <= .5)),
                           ('confident_shared_interior', shared & (confidence > .5)),
                           ('unshared_or_edge', ~shared)]:
            result[name]['matched'] += int(labels[mask].sum())
            result[name]['unmatched'] += int((1 - labels[mask]).sum())
    return result


def select(items, context, acoustic):
    best, search, memo = None, [], {}
    for guardian in GUARDIAN_THRESHOLDS:
        for threshold in THRESHOLDS:
            raw, margin = [score(items, [np.where((c < at) & (a < g), 0., 1.)
                           for c, a in zip(context, acoustic, strict=True)], at, memo, ceiling=1.)
                           for at, g in ((threshold, guardian), (threshold * .5, guardian * .5))]
            search.append({'guardian_raw': guardian,
                           'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                           'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0 and (
                    best is None or margin['false_notes_removed'] > best[0]['false_notes_removed']):
                best = margin, guardian * .5
    return best, search


def choose(rows):
    eligible = [row for row in rows if row['selected'] and row['validation_false_notes_removed'] > 0]
    # Plan order is the fixed tie breaker; no test metrics participate.
    return max(eligible, key=lambda row: row['validation_false_notes_removed']) if eligible else None


def train(directory, output, extras):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {'profiles': list(PROFILES), 'baseline_hashes': hashes(), 'policy': POLICY,
            'ceiling': 1., 'pitch_range': [36, 96], 'edge_seconds': 2.5,
            'threshold_grid': THRESHOLDS, 'guardian_threshold_grid': GUARDIAN_THRESHOLDS,
            'selection_safety_factor': .5, 'sklearn_version': sklearn.__version__,
            'extra_manifest_sha256': {str(path): digest(path) for path in extras},
            'selection': 'Validation-only maximum gated gain, fixed profile-order tie break; one regression winner',
            'preservation': 'Every matched attack and hold, no per-recording false-note increase',
            'scope': 'Existing licensed cached dataset events only; no user audio or scores; offline policy'}
    write(output / 'plan.json', plan)
    training, validation = collect(directory, extras)
    write(output / 'label-audit.json', audit(training, validation))
    print('Preparing pinned V7 training/validation evidence', flush=True)
    training, validation = [prepare(directory, items, baseline_evidence=True, full_register=True)
                            for items in (training, validation)]
    write(output / 'coverage.json', {'validation_only': True, 'clips': len(validation), 'counts': coverage(validation)})
    vx, vy, vw, _ = matrix(validation, 1.)
    rows = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        settings = {k: v for k, v in profile.items() if k not in ('name', 'positive_weight')}
        settings.update(learning_rate=.04, early_stopping=False, random_state=261020 + index * 2)
        guardian_settings = {**settings, 'max_leaf_nodes': 15, 'random_state': 261021 + index * 2}
        write(destination / 'run.json', {'settings': settings, 'guardian_settings': guardian_settings,
              'positive_weight': profile['positive_weight'], 'training_counts': counts,
              'training_clips': len(training), 'validation_clips': len(validation), 'training_events': len(y),
              'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, curves = [], {}
        for name, config, tx, test_x in [('context', settings, x, vx),
                                       ('guardian', guardian_settings, acoustic_view(x), acoustic_view(vx))]:
            print(f'Fitting {profile["name"]} {name}: {len(y)} events, {config["max_iter"]} trees', flush=True)
            model = HistGradientBoostingClassifier(**config).fit(tx, y, sample_weight=weight)
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, prediction,
                            sample_weight=vw, labels=[0., 1.]))}
                            for step, prediction in enumerate(model.staged_predict_proba(test_x), 1)
                            if step % 50 == 0 or step == config['max_iter']]
            models.append(model)
            write(destination / 'learning-curves.json', curves)
        context = [models[0].predict_proba(item['x'])[:, 1] for item in validation]
        acoustic = [models[1].predict_proba(acoustic_view(item['x']))[:, 1] for item in validation]
        best, search = select(validation, context, acoustic)
        result, guardian = best if best else (score(validation, [np.ones(len(item['events']))
                                                               for item in validation], 0., ceiling=1.), 0.)
        with (destination / 'candidate.pickle').open('wb') as stream:
            pickle.dump(models, stream)
        selection = {'name': profile['name'], 'selected': best is not None,
                     'validation_false_notes_removed': result['false_notes_removed'],
                     'threshold': result['threshold'], 'guardian_threshold': guardian,
                     'checkpoint_sha256': digest(destination / 'candidate.pickle'),
                     'plan_sha256': digest(output / 'plan.json'), 'policy': POLICY, 'ceiling': 1.,
                     'baseline_hashes': hashes(), 'test_not_evaluated': True}
        write(destination / 'validation.json', result)
        write(destination / 'search.json', search)
        write(destination / 'selection.json', selection)
        rows.append(selection)
        print(json.dumps(selection), flush=True)
    winner = choose(rows)
    write(output / 'batch-selection.json', {'selected': winner is not None, 'winner': winner,
          'candidates': rows, 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
    print(json.dumps({'batch_winner': winner}, indent=2), flush=True)


def load_winner(output):
    batch = json.loads((output / 'batch-selection.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    winner = batch['winner']
    if (not batch['selected'] or winner != choose(batch['candidates'])
            or batch['plan_sha256'] != digest(output / 'plan.json')
            or plan['policy'] != POLICY or plan['ceiling'] != 1.
            or plan['profiles'] != list(PROFILES) or plan['selection_safety_factor'] != .5
            or plan['pitch_range'] != [36, 96] or plan['edge_seconds'] != 2.5
            or plan['threshold_grid'] != list(THRESHOLDS)
            or plan['guardian_threshold_grid'] != list(GUARDIAN_THRESHOLDS)
            or plan['baseline_hashes'] != hashes() or winner['baseline_hashes'] != hashes()
            or winner['name'] not in [p['name'] for p in plan['profiles']]
            or winner['plan_sha256'] != batch['plan_sha256']
            or winner['policy'] != POLICY or winner['ceiling'] != 1.
            or winner['threshold'] not in {v * .5 for v in plan['threshold_grid'] if v > 0}
            or winner['guardian_threshold'] not in {v * .5 for v in plan['guardian_threshold_grid']}):
        raise ValueError('Ineligible or changed frozen batch selection')
    source = output / winner['name']
    if json.loads((source / 'selection.json').read_text()) != winner:
        raise ValueError('Changed frozen candidate selection')
    for filename, expected in plan['extra_manifest_sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Changed fitting/validation manifest')
    checkpoint = source / 'candidate.pickle'
    if digest(checkpoint) != winner['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    if len(models) != 2 or [m.n_features_in_ for m in models] != [74, 54]:
        raise ValueError('Unexpected model feature contract')
    return winner, models


def test(directory, output, fresh=None):
    destination = output / ('fresh.json' if fresh else 'regression.json')
    if destination.exists():
        raise ValueError('Preserve consumed evaluation; do not retune on regression')
    winner, models = load_winner(output)
    if fresh:
        prior = json.loads((output / 'regression.json').read_text())
        if not prior['passes'] or prior['false_notes_removed'] <= 0:
            raise ValueError('Fresh evaluation requires positive regression improvement')
        items = list(cached_external(directory, fresh, directory))
        if not items or any(item['group'] != 'test' for item in items):
            raise ValueError('Fresh evaluation requires reserved test performers')
    else:
        items = regression_items(directory)
    torch.set_num_threads(2)
    items = prepare(directory, items, baseline_evidence=True, full_register=True)
    result = score(items, [probabilities(models, item['x'], winner['threshold'], winner['guardian_threshold'])
                           for item in items], winner['threshold'], ceiling=1.)
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(output / 'batch-selection.json'),
                  selection_sha256=digest(output / winner['name'] / 'selection.json'),
                  guardian_threshold=winner['guardian_threshold'], policy=POLICY,
                  scope='Later passages, same reserved performers/compositions' if fresh
                  else 'Consumed regression; not independent generalization evidence')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='broad-consensus-v9')
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    if args.test or args.fresh:
        test(args.directory, args.directory / args.run, args.fresh)
    else:
        train(args.directory, args.directory / args.run, args.extra)
