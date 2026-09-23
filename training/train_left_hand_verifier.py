"""Fit a low-register accompaniment specialist using cached labeled data only.

The bass stem has a different decoder and is deliberately outside this model's
scope. Freeze selection before regression; reject loss of correct held notes as
well as attacks. Never process uploads or generate scores in this workflow.
"""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import sklearn
import torch
from app.services.accompaniment_verifier import AccompanimentVerifier
from app.services.note_context_model import ContextNoteModel
from audit_verifier_labels import audit
from cache_note_verifier import targets
from export_context_verifier import export
from sklearn.ensemble import HistGradientBoostingClassifier
from train_note_verifier import load_data, write
from train_relational_verifier import hashes
from train_residual_verifier import (
    THRESHOLDS,
    cached_external,
    evaluate,
    prepare_items,
    residual_keep,
)


def prepare(directory, items, verifier):
    prepared = prepare_items(directory, items, verifier)
    for item in prepared:
        item['keep'] = residual_keep(item, verifier.residual_model.probability(item['x']),
                                     verifier.residual_model.threshold)
        # Match the website's bass-staff split, without learning absolute pitch
        # or changing any treble/melody or independent bass-stem decisions.
        item['shared'] &= (item['events'][:, 2] >= 36) & (item['events'][:, 2] < 60)
    return prepared


def held_matches(reference, events):
    return {i for i, _ in mir_eval.transcription.match_notes(
        reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]),
        onset_tolerance=.05, offset_ratio=.2, offset_min_tolerance=.05)}


def score(items, probabilities, threshold):
    result = evaluate(items, probabilities, threshold)
    for item, probability, row in zip(items, probabilities, result['per_recording'], strict=True):
        before = item['events'][item['keep']]
        after = item['events'][residual_keep(item, probability, threshold)]
        preserved = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        row['held_references_preserved'] = preserved
        row['passes'] &= preserved
        row['deployed_v4'] = row.pop('deployed_v3')
    for row in result['aggregate'].values():
        row['deployed_v4'] = row.pop('deployed_v3')
    result['passes'] = all(row['passes'] for row in result['per_recording'])
    result['failed_recordings'] = [row['id'] for row in result['per_recording'] if not row['passes']]
    return result


def train(directory, output, extra=None):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    if extra:
        for item in cached_external(directory, extra, directory):
            if item['group'] not in ('train', 'validation'):
                raise ValueError('Test data cannot enter fitting or selection')
            item['y'], item['mask'] = targets(item['reference'], item['events'])
            (training if item['group'] == 'train' else validation).append(item)
    write(output / 'label-audit.json', audit(training, validation))
    verifier = AccompanimentVerifier()
    training, validation = [prepare(directory, items, verifier) for items in (training, validation)]
    all_x, all_y, weights, counts = [], [], [], []
    for corpus in sorted({item['corpus'] for item in training}):
        records = [item for item in training if item['corpus'] == corpus]
        for item in records:
            mask = item['mask'] & item['keep'] & item['shared']
            x, y = item['x'][mask], item['y'][mask]
            if not len(y):
                continue
            all_x.append(x)
            all_y.append(y)
            weights.append(np.where(y == 1, 8., 1.) / (len(records) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus, 'positive': int(y.sum()),
                           'negative': int(len(y) - y.sum())})
    x, y, weight = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(weights)
    weight /= weight.mean()
    settings = {'max_iter': 300, 'max_leaf_nodes': 15, 'min_samples_leaf': 40,
                'learning_rate': .05, 'l2_regularization': 4.,
                'early_stopping': False, 'random_state': 260925}
    write(output / 'run.json', {'settings': settings, 'positive_weight': 8.,
        'training_clips': len(training), 'validation_clips': len(validation),
        'training_events': len(y), 'training_counts': counts, 'baseline_hashes': hashes(),
        'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5, 'ceiling': .5,
        'pitch_range': [36, 60], 'pitch_upper_exclusive': True,
        'sklearn_version': sklearn.__version__, 'test_used_for_selection': False,
        'extra_manifest_sha256': hashlib.sha256(extra.read_bytes()).hexdigest() if extra else None,
        'scope': '52 cached acoustic features; low accompaniment only; no bass-stem training',
        'preservation': 'Every matched onset and onset-offset reference in each recording'})
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight)
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
    export(model, threshold, output / 'candidate.npz', validation)
    write(output / 'validation.json', result)
    write(output / 'search.json', search)
    selection = {'selected': bool(best and result['passes'] and result['false_notes_removed'] > 0),
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.npz').read_bytes()).hexdigest(),
        'threshold': threshold, 'ceiling': .5, 'baseline_hashes': hashes(),
        'pitch_range': [36, 60], 'test_not_evaluated': True,
        'selection': 'Validation preservation of attacks and holds; halve threshold before regression'}
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


def test(directory, output):
    destination = output / 'regression.json'
    if destination.exists():
        raise ValueError('Preserve consumed results; do not retune on regression')
    selection = json.loads((output / 'selection.json').read_text())
    if not selection['selected'] or selection['baseline_hashes'] != hashes():
        raise ValueError('Ineligible candidate or changed baseline')
    checkpoint = output / 'candidate.npz'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with np.load(checkpoint, allow_pickle=False) as saved:
        model = ContextNoteModel(saved)
    if model.threshold != selection['threshold'] or selection['ceiling'] != .5 or selection['pitch_range'] != [36, 60]:
        raise ValueError('Candidate policy changed after freeze')
    items = load_data(directory, 'test')
    for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                 'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails'):
        base = directory.parent if name.startswith('note-verifier-') else directory
        items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    items = prepare(directory, items, AccompanimentVerifier())
    result = score(items, [model.probability(item['x']) for item in items], model.threshold)
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  scope='Consumed regression excerpts; not independent generalization evidence')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='left-hand-verifier-v1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--extra', type=Path)
    args = parser.parse_args()
    output = args.directory / args.run
    if args.test:
        test(args.directory, output)
    else:
        train(args.directory, output, args.extra)
