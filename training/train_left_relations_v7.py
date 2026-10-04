"""Learn remaining V6 low-note errors from complete neighboring-note context."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
import soundfile as sf
import torch
from app.services.accompaniment_verifier import (
    CHECKPOINT,
    AccompanimentVerifier,
)
from app.services.note_context_model import ContextNoteModel
from audit_verifier_labels import audit
from cache_note_verifier import targets
from evaluate_context_correction import matched_references
from note_relations import (
    ALL_NAMES,
    EDGE_SECONDS,
    RELATION_VERSION,
    relation_features,
    window_eligible,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from train_left_hand_verifier import held_matches
from train_left_hand_verifier import prepare as prepare_v4
from train_note_verifier import aggregate, load_data, metrics, write
from train_relational_verifier import hashes as v4_hashes
from train_residual_verifier import THRESHOLDS, cached_external, residual_keep

BASELINE_LEFT_HAND = CHECKPOINT.parent / 'accompaniment-left-hand-v1.npz'
BASELINE_SHA256 = '7b3343637fa0d19ba1cd468845fcc309f30ac55b7fddc00f68bbc607abc2555b'
BASELINE_REFINEMENT = CHECKPOINT.parent / 'accompaniment-left-refinement-v3.npz'
REFINEMENT_SHA256 = '6b0f9bb5260f9645831dd409abd6fcb17cd8acbdd09e7a13df9c356cd76414e2'


def hashes():
    digest = hashlib.sha256(BASELINE_LEFT_HAND.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise ValueError('Frozen website V5 baseline changed')
    refinement_digest = hashlib.sha256(BASELINE_REFINEMENT.read_bytes()).hexdigest()
    if refinement_digest != REFINEMENT_SHA256:
        raise ValueError('Frozen website V6 baseline changed')
    return {**v4_hashes(), BASELINE_LEFT_HAND.name: digest, BASELINE_REFINEMENT.name: refinement_digest}


def complete_evidence(directory, item):
    """Return full legacy geometry; never silently use an incomplete large crop."""
    start = item.get('evaluation_window', [0.])[0]
    if 'legacy_x' in item and start <= .25:
        return item['legacy_events'], item['legacy_x']
    path = directory.parent / 'note-verifier-features' / (item.get('id', '') + '.npz')
    if not path.is_file():
        raise ValueError('A large cropped legacy cache needs full neighboring context')
    with np.load(path, allow_pickle=False) as saved:
        return saved['events'], saved['x']


def enrich(item, events, x, probability, duration):
    """Score target features using neighbors outside the annotation crop too."""
    keep = (events[:, 2] >= 36) & (events[:, 2] < 96)
    events, x, probability = events[keep], x[keep], probability[keep]
    lookup = {tuple(event): i for i, event in enumerate(events)}
    indices = np.asarray([lookup[tuple(event)] for event in item['events']], dtype=int)
    # Preserve unshared original acoustic columns too; they still influence V2.
    np.testing.assert_allclose(probability[indices], item['p'], rtol=1e-5, atol=1e-7)
    full_x = np.zeros((len(events), 52), dtype=np.float32)
    full_x[:, :26] = x
    full_x[indices] = item['x']
    features = relation_features(events, full_x, probability)[indices]
    return features, window_eligible(item['events'], duration)


def prepare(directory, items, policy='refinement'):
    if policy != 'refinement':
        raise ValueError('V7 preserves all V6 rejections')
    hashes()
    verifier = AccompanimentVerifier()
    with np.load(BASELINE_LEFT_HAND, allow_pickle=False) as saved:
        baseline = ContextNoteModel(saved)
    with np.load(BASELINE_REFINEMENT, allow_pickle=False) as saved:
        refinement = ContextNoteModel(saved)
    prepared = prepare_v4(directory, items, verifier)
    for item in prepared:
        item['keep'] = residual_keep(item, baseline.probability(item['x']), baseline.threshold)
        item['keep'] = residual_keep(item, refinement.probability(item['x']), refinement.threshold)
        item['baseline_keep'] = item['keep'].copy()
        retained_y, retained_mask = targets(item['reference'], item['events'][item['keep']])
        item['y'], item['mask'] = np.zeros(len(item['events'])), np.zeros(len(item['events']), dtype=bool)
        item['y'][item['keep']], item['mask'][item['keep']] = retained_y, retained_mask
        events, x = complete_evidence(directory, item)
        with torch.inference_mode():
            probability = torch.sigmoid(verifier.model(torch.from_numpy(
                (x - verifier.mean) / verifier.scale))).numpy()
        duration = min(30., sf.info(directory / item['audio']).duration)
        item['x'], eligible = enrich(item, events, x, probability, duration)
        item['shared'] &= eligible
    return prepared


def score(items, probabilities, threshold, memo=None):
    rows = []
    for item, probability in zip(items, probabilities, strict=True):
        before = item['events'][item['baseline_keep']]
        after_mask = residual_keep(item, probability, threshold)
        cache_key = (id(item), after_mask.tobytes())
        if memo is not None and cache_key in memo:
            rows.append(memo[cache_key])
            continue
        after = item['events'][after_mask]
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'deployed_v6': old, 'candidate': new,
                     'matched_references_preserved': attacks, 'held_references_preserved': holds,
                     'passes': attacks and holds and new['false_positives'] <= old['false_positives']})
        if memo is not None:
            memo[cache_key] = rows[-1]
    return {'threshold': threshold, 'passes': all(r['passes'] for r in rows),
            'false_notes_removed': sum(r['deployed_v6']['false_positives'] - r['candidate']['false_positives'] for r in rows),
            'failed_recordings': [r['id'] for r in rows if not r['passes']],
            'aggregate': {corpus: {system: aggregate([r[system] for r in rows if r['corpus'] == corpus])
                                  for system in ('deployed_v6', 'candidate')}
                          for corpus in sorted({r['corpus'] for r in rows})}, 'per_recording': rows}


def combine_scores(context, acoustic, context_threshold, acoustic_threshold):
    if not 0 <= context_threshold <= 1 or not 0 < acoustic_threshold <= 1:
        raise ValueError('Invalid consensus thresholds')
    return np.maximum(context, np.minimum(1., acoustic * context_threshold / acoustic_threshold))


def predict(model, x, *, threshold=None, guardian_threshold=None):
    """Consensus accepts a note if either independently fitted view accepts it."""
    if isinstance(model, list):
        if guardian_threshold is not None:
            context = model[0][1].predict_proba(x)[:, 1]
            acoustic = model[1][1].predict_proba(x[:, :52])[:, 1]
            return combine_scores(context, acoustic, threshold, guardian_threshold)
        return np.maximum.reduce([head.predict_proba(x[:, :width])[:, 1] for width, head in model])
    return model.predict_proba(x)[:, 1]


def train(directory, output, extras, leaves=31, policy='refinement', consensus=False):
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
    settings = {'max_iter': 300, 'max_leaf_nodes': leaves, 'min_samples_leaf': 40,
                'learning_rate': .05, 'l2_regularization': 4.,
                'early_stopping': False, 'random_state': 261004 if consensus else 261003}
    acoustic_settings = {**settings, 'max_leaf_nodes': 15, 'random_state': 261005}
    write(output / 'run.json', {'settings': settings, 'positive_weight': 8.,
        'training_clips': len(training), 'validation_clips': len(validation),
        'training_events': len(y), 'training_counts': counts, 'baseline_hashes': hashes(),
        'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5, 'ceiling': .5,
        'pitch_range': [36, 60], 'pitch_upper_exclusive': True,
        'sklearn_version': sklearn.__version__, 'test_used_for_selection': False,
        'extra_manifest_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extras},
        'policy': policy,
        'feature_names': ALL_NAMES, 'feature_version': RELATION_VERSION, 'edge_seconds': EDGE_SECONDS,
        'consensus': consensus, 'acoustic_settings': acoustic_settings if consensus else None,
        'scope': 'V6 low-register correction; complete context; no bass-stem or user audio',
        'preservation': 'Every matched onset and onset-offset reference; no per-recording false-note increase'})
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight)
    if consensus:
        acoustic_model = HistGradientBoostingClassifier(**acoustic_settings).fit(x[:, :52], y, sample_weight=weight)
        model = [(72, model), (52, acoustic_model)]
    probabilities = [predict(model, item['x']) for item in validation]
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
    with (output / 'candidate.pickle').open('wb') as stream:
        pickle.dump(model, stream)  # Local research artifact; never loaded by website.
    write(output / 'validation.json', result)
    write(output / 'search.json', search)
    selection = {'selected': best is not None,
        'policy': policy,
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.pickle').read_bytes()).hexdigest(),
        'threshold': threshold, 'ceiling': .5, 'baseline_hashes': hashes(),
        'pitch_range': [36, 60], 'test_not_evaluated': True,
        'feature_version': RELATION_VERSION, 'edge_seconds': EDGE_SECONDS,
        'consensus': consensus,
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
    if (selection['policy'] != 'refinement' or selection['policy'] != run['policy']
            or selection['feature_version'] != RELATION_VERSION or selection['edge_seconds'] != EDGE_SECONDS
            or selection.get('consensus', False) != run.get('consensus', False)):
        raise ValueError('Candidate policy changed after freeze')
    if selection.get('calibration') != run.get('calibration'):
        raise ValueError('Candidate calibration changed after freeze')
    if fresh:
        regression = output / 'regression.json'
        if not regression.exists() or not json.loads(regression.read_text())['passes']:
            raise ValueError('Preserve fresh evaluation until regression passes')
    checkpoint = output / 'candidate.pickle'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with checkpoint.open('rb') as stream:
        model = pickle.load(stream)  # Hash-verified locally fitted artifact.
    if selection['ceiling'] != .5 or selection['pitch_range'] != [36, 60]:
        raise ValueError('Candidate policy changed after freeze')
    if fresh:
        items = list(cached_external(directory, fresh, directory))
        if not items or any(item.get('group') != 'test' for item in items):
            raise ValueError('Fresh evaluation requires reserved test performers')
    else:
        items = load_data(directory, 'test')
        for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                     'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails', 'relational-piano-fresh'):
            base = directory.parent if name.startswith('note-verifier-') else directory
            items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    items = prepare(directory, items, selection['policy'])
    result = score(items, [predict(model, item['x'], threshold=selection['threshold'],
                                  guardian_threshold=selection.get('guardian_threshold')) for item in items],
                   selection['threshold'])
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  scope='Unscored later passages; same reserved test performers and compositions' if fresh
                  else 'Consumed regression excerpts; not independent generalization evidence')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='left-relations-v7')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--leaves', type=int, choices=(15, 31), default=31)
    parser.add_argument('--policy', choices=('refinement',), default='refinement')
    parser.add_argument('--consensus', action='store_true')
    args = parser.parse_args()
    output = args.directory / args.run
    if args.test or args.fresh:
        test(args.directory, output, args.fresh)
    else:
        train(args.directory, output, args.extra, args.leaves, args.policy, args.consensus)
