"""Fit a replacement or residual low-register head on licensed dataset passages.

Compare with frozen website V5, including its existing left-hand head. Do not
claim earlier improvements again or tune on regression. Refinement mode freezes
the existing left-hand decisions and learns only remaining candidate errors.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from app.services.accompaniment_verifier import CHECKPOINT, AccompanimentVerifier
from app.services.note_context_model import ContextNoteModel
from audit_verifier_labels import audit
from cache_note_verifier import targets
from evaluate_context_correction import matched_references
from export_context_verifier import export
from sklearn.ensemble import HistGradientBoostingClassifier
from train_left_hand_verifier import held_matches
from train_left_hand_verifier import prepare as prepare_v4
from train_note_verifier import aggregate, load_data, metrics, write
from train_relational_verifier import hashes as v4_hashes
from train_residual_verifier import THRESHOLDS, cached_external, residual_keep

BASELINE_LEFT_HAND = CHECKPOINT.parent / 'accompaniment-left-hand-v1.npz'
BASELINE_SHA256 = '7b3343637fa0d19ba1cd468845fcc309f30ac55b7fddc00f68bbc607abc2555b'


def hashes():
    digest = hashlib.sha256(BASELINE_LEFT_HAND.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise ValueError('Frozen website V5 baseline changed')
    return {**v4_hashes(), BASELINE_LEFT_HAND.name: digest}


def prepare(directory, items, policy='replacement'):
    if policy not in ('replacement', 'refinement'):
        raise ValueError('Unknown low-register model policy')
    hashes()
    verifier = AccompanimentVerifier()
    with np.load(BASELINE_LEFT_HAND, allow_pickle=False) as saved:
        baseline = ContextNoteModel(saved)
    prepared = prepare_v4(directory, items, verifier)
    for item in prepared:
        # Keep is the common V4 input to either old or replacement left-hand head.
        item['baseline_keep'] = residual_keep(item, baseline.probability(item['x']), baseline.threshold)
        if policy == 'refinement':
            item['keep'] = item['baseline_keep'].copy()
            # Re-match after filtering: if a duplicate was removed, its surviving
            # counterpart may now be the correct positive for this reference.
            retained_y, retained_mask = targets(item['reference'], item['events'][item['keep']])
            item['y'], item['mask'] = np.zeros(len(item['events'])), np.zeros(len(item['events']), dtype=bool)
            item['y'][item['keep']], item['mask'][item['keep']] = retained_y, retained_mask
    return prepared


def score(items, probabilities, threshold):
    rows = []
    for item, probability in zip(items, probabilities, strict=True):
        before = item['events'][item['baseline_keep']]
        after = item['events'][residual_keep(item, probability, threshold)]
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'deployed_v5': old, 'candidate': new,
                     'matched_references_preserved': attacks, 'held_references_preserved': holds,
                     'passes': attacks and holds and new['false_positives'] <= old['false_positives']})
    return {'threshold': threshold, 'passes': all(r['passes'] for r in rows),
            'false_notes_removed': sum(r['deployed_v5']['false_positives'] - r['candidate']['false_positives'] for r in rows),
            'failed_recordings': [r['id'] for r in rows if not r['passes']],
            'aggregate': {corpus: {system: aggregate([r[system] for r in rows if r['corpus'] == corpus])
                                  for system in ('deployed_v5', 'candidate')}
                          for corpus in sorted({r['corpus'] for r in rows})}, 'per_recording': rows}


def train(directory, output, extras, leaves=15, policy='replacement'):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    identities = {item['id'] for item in training + validation}
    for extra in extras:
        for item in cached_external(directory, extra, directory):
            if item['group'] not in ('train', 'validation') or item['id'] in identities:
                raise ValueError('Test data or duplicated passages cannot enter fitting/selection')
            identities.add(item['id'])
            item['y'], item['mask'] = targets(item['reference'], item['events'])
            (training if item['group'] == 'train' else validation).append(item)
    write(output / 'label-audit.json', audit(training, validation))
    training, validation = [prepare(directory, items, policy) for items in (training, validation)]
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
    # The first run keeps v1 capacity; an explicit larger-capacity validation
    # experiment is recorded separately. No regression results select capacity.
    settings = {'max_iter': 300, 'max_leaf_nodes': leaves, 'min_samples_leaf': 40,
                'learning_rate': .05, 'l2_regularization': 4.,
                'early_stopping': False, 'random_state': 260925}
    write(output / 'run.json', {'settings': settings, 'positive_weight': 8.,
        'training_clips': len(training), 'validation_clips': len(validation),
        'training_events': len(y), 'training_counts': counts, 'baseline_hashes': hashes(),
        'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5, 'ceiling': .5,
        'pitch_range': [36, 60], 'pitch_upper_exclusive': True,
        'sklearn_version': sklearn.__version__, 'test_used_for_selection': False,
        'extra_manifest_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extras},
        'policy': policy,
        'scope': 'V5 low-register correction; 52 acoustic features; no bass-stem or user audio',
        'preservation': 'Every matched onset and onset-offset reference; no per-recording false-note increase'})
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight)
    probabilities = [model.predict_proba(item['x'])[:, 1] for item in validation]
    best, search = None, []
    for threshold in THRESHOLDS:
        selected, margin = [score(validation, probabilities, value) for value in (threshold, threshold * .5)]
        search.append({'raw': {k: v for k, v in selected.items() if k != 'per_recording'},
                       'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
        if selected['passes'] and margin['passes'] and margin['false_notes_removed'] > 0 and (
                best is None or margin['false_notes_removed'] > best['false_notes_removed']):
            best = margin
    threshold = best['threshold'] if best else 0.
    result = best or score(validation, probabilities, threshold)
    export(model, threshold, output / 'candidate.npz', validation)
    write(output / 'validation.json', result)
    write(output / 'search.json', search)
    selection = {'selected': best is not None,
        'policy': policy,
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.npz').read_bytes()).hexdigest(),
        'threshold': threshold, 'ceiling': .5, 'baseline_hashes': hashes(),
        'pitch_range': [36, 60], 'test_not_evaluated': True,
        'selection': 'Best validation gain passing attack/hold/false-note gates at raw and half-margin threshold'}
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


def test(directory, output, fresh=None):
    destination = output / ('fresh.json' if fresh else 'regression.json')
    if destination.exists():
        raise ValueError('Preserve consumed results; do not retune on regression')
    selection = json.loads((output / 'selection.json').read_text())
    if not selection['selected'] or selection['baseline_hashes'] != hashes():
        raise ValueError('Ineligible candidate or changed baseline')
    run = json.loads((output / 'run.json').read_text())
    if selection.get('policy', 'replacement') != run.get('policy', 'replacement'):
        raise ValueError('Candidate policy changed after freeze')
    if fresh:
        regression = output / 'regression.json'
        if not regression.exists() or not json.loads(regression.read_text())['passes']:
            raise ValueError('Preserve fresh evaluation until regression passes')
    checkpoint = output / 'candidate.npz'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with np.load(checkpoint, allow_pickle=False) as saved:
        model = ContextNoteModel(saved)
    if model.threshold != selection['threshold'] or selection['ceiling'] != .5 or selection['pitch_range'] != [36, 60]:
        raise ValueError('Candidate policy changed after freeze')
    if fresh:
        items = list(cached_external(directory, fresh, directory))
        if not items or any(item.get('group') != 'test' for item in items):
            raise ValueError('Fresh evaluation requires reserved test performers')
    else:
        items = load_data(directory, 'test')
        for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                     'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails'):
            base = directory.parent if name.startswith('note-verifier-') else directory
            items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    items = prepare(directory, items, selection.get('policy', 'replacement'))
    result = score(items, [model.probability(item['x']) for item in items], model.threshold)
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  scope='Unscored later passages; same reserved test performers and compositions' if fresh
                  else 'Consumed regression excerpts; not independent generalization evidence')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='left-hand-expanded-v2')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--leaves', type=int, choices=(15, 31), default=15)
    parser.add_argument('--policy', choices=('replacement', 'refinement'), default='replacement')
    args = parser.parse_args()
    output = args.directory / args.run
    if args.test or args.fresh:
        test(args.directory, output, args.fresh)
    else:
        train(args.directory, output, args.extra, args.leaves, args.policy)
