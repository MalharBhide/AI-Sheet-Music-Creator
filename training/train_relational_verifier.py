"""Train a relationship-aware false-note candidate against frozen deployed V4.

This experiment stays offline until all preservation gates pass. No user audio
or sheet-music generation is needed: all inputs are cached corpus features.
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
from note_relations import ALL_NAMES, RELATION_VERSION, complete_context, relation_features
from sklearn.ensemble import HistGradientBoostingClassifier
from train_note_verifier import load_data, write
from train_residual_verifier import (
    THRESHOLDS,
    baseline_hashes,
    cached_external,
    evaluate,
    prepare_items,
    residual_keep,
)

from app.services.accompaniment_verifier import RESIDUAL_CHECKPOINT, AccompanimentVerifier


def hashes():
    return {**baseline_hashes(), RESIDUAL_CHECKPOINT.name:
            hashlib.sha256(RESIDUAL_CHECKPOINT.read_bytes()).hexdigest()}


def prepare(directory, items, verifier):
    prepared = prepare_items(directory, items, verifier)
    for item in prepared:
        probability = verifier.residual_model.probability(item['x'])
        item['keep'] = residual_keep(item, probability, verifier.residual_model.threshold)
        # Legacy candidates outside the shared decoder remain protected. All
        # neighboring candidate geometry/confidence is label-free evidence.
        item['x'] = relation_features(item['events'], item['x'], item['p'])
        # Cropped caches lack neighbors outside their evaluation window. Protect
        # edge targets instead of teaching/deploying mismatched features there.
        start, stop = item.get('evaluation_window', (0., float(item['seconds'])))
        item['shared'] = item['shared'] & complete_context(item['events'], start, stop)
    return prepared


def score(items, probabilities, threshold):
    result = evaluate(items, probabilities, threshold)
    # The inherited metric name describes its older experiment, not this baseline.
    for row in result['per_recording']:
        row['deployed_v4'] = row.pop('deployed_v3')
    for row in result['aggregate'].values():
        row['deployed_v4'] = row.pop('deployed_v3')
    return result


def train(directory, output, extra=None):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    if extra:
        additions = list(cached_external(directory, extra, directory))
        for item in additions:
            if item['group'] not in ('train', 'validation'):
                raise ValueError('Test data cannot enter training or selection')
            item['y'], item['mask'] = targets(item['reference'], item['events'])
            (training if item['group'] == 'train' else validation).append(item)
    write(output / 'label-audit.json', audit(training, validation))
    verifier = AccompanimentVerifier()
    training, validation = [prepare(directory, items, verifier) for items in (training, validation)]
    all_x, all_y, weights, counts = [], [], [], []
    for corpus in sorted({i['corpus'] for i in training}):
        records = [i for i in training if i['corpus'] == corpus]
        for item in records:
            mask = item['mask'] & item['keep'] & item['shared']
            x, y = item['x'][mask], item['y'][mask]
            if not len(y):
                continue
            all_x.append(x)
            all_y.append(y)
            weights.append(np.where(y == 1, 6., 1.) / (len(records) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus, 'positive': int(y.sum()),
                           'negative': int(len(y) - y.sum())})
    x, y, weight = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(weights)
    weight /= weight.mean()
    settings = {'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 40,
                'learning_rate': .05, 'l2_regularization': 3.,
                'early_stopping': False, 'random_state': 260923}
    write(output / 'run.json', {'settings': settings, 'positive_weight': 6.,
        'training_clips': len(training), 'validation_clips': len(validation),
        'training_events': len(y), 'training_counts': counts, 'baseline_hashes': hashes(),
        'feature_names': ALL_NAMES, 'feature_version': RELATION_VERSION,
        'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5, 'ceiling': .5,
        'sklearn_version': sklearn.__version__, 'test_used_for_selection': False,
        'extra_manifest_sha256': hashlib.sha256(extra.read_bytes()).hexdigest() if extra else None,
        'scope': 'Cached licensed corpus features only; protect every matched reference note',
        'weighting': 'Equal corpus/recording base weight, then sixfold positive weight'})
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight)
    with (output / 'candidate.pickle').open('wb') as stream:
        pickle.dump(model, stream)  # Local research artifact only; never deployed.
    probabilities = [model.predict_proba(item['x'])[:, 1] for item in validation]
    best, search = None, []
    for threshold in THRESHOLDS:
        result = score(validation, probabilities, threshold)
        search.append({k: v for k, v in result.items() if k != 'per_recording'})
        if result['passes'] and result['false_notes_removed'] > 0 and (
                best is None or result['false_notes_removed'] > best['false_notes_removed']):
            best = result
    threshold = best['threshold'] * .5 if best else 0.
    result = score(validation, probabilities, threshold)
    write(output / 'validation.json', result)
    write(output / 'search.json', search)
    selection = {'selected': bool(best and result['passes'] and result['false_notes_removed'] > 0),
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.pickle').read_bytes()).hexdigest(),
        'threshold': threshold, 'ceiling': .5, 'baseline_hashes': hashes(),
        'feature_version': RELATION_VERSION, 'test_not_evaluated': True,
        'selection': 'Most validation false notes removed while preserving every matched reference, then halve threshold before test'}
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


def test(directory, output, fresh=None):
    destination = output / ('fresh.json' if fresh else 'regression.json')
    if destination.exists():
        raise ValueError('Preserve consumed results; do not tune on this regression')
    selection = json.loads((output / 'selection.json').read_text())
    if not selection['selected'] or selection['baseline_hashes'] != hashes():
        raise ValueError('Ineligible candidate or changed baseline')
    if selection['feature_version'] != RELATION_VERSION or selection['ceiling'] != .5:
        raise ValueError('Feature contract or decision policy changed after freeze')
    if fresh:
        regression = output / 'regression.json'
        if not regression.exists() or not json.loads(regression.read_text())['passes']:
            raise ValueError('Preserve unused evaluation data until the regression gate passes')
    checkpoint = output / 'candidate.pickle'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with checkpoint.open('rb') as stream:
        model = pickle.load(stream)  # Checksum-verified local training output.
    if fresh:
        items = list(cached_external(directory, fresh, directory))
        if any(item['group'] != 'test' for item in items):
            raise ValueError('Fresh evaluation requires reserved test performers')
    else:
        items = load_data(directory, 'test')
        for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                     'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails'):
            base = directory.parent if name.startswith('note-verifier-') else directory
            items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    items = prepare(directory, items, AccompanimentVerifier())
    result = score(items, [model.predict_proba(item['x'])[:, 1] for item in items], selection['threshold'])
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  scope='Unscored later piano passages; same test performers/compositions' if fresh
                  else '142 consumed regression excerpts; not fresh test evidence')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='relational-verifier-v1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--extra', type=Path)
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    if args.test or args.fresh:
        test(args.directory, args.directory / args.run, args.fresh)
    else:
        train(args.directory, args.directory / args.run, args.extra)
