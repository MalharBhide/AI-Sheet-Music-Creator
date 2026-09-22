"""Learn remaining false notes from cached training data, against deployed V3.

No user audio, source separation, score rendering or website jobs are used.
The fixed candidate is selected on validation and evaluated once after freeze.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from audit_verifier_labels import audit
from cache_note_verifier import targets
from evaluate_context_correction import matched_references, previous_events
from export_context_verifier import export
from sklearn.ensemble import HistGradientBoostingClassifier
from train_note_verifier import aggregate, load_data, metrics, write

from app.services.accompaniment_verifier import (
    CHECKPOINT,
    CONTEXT_CHECKPOINTS,
    AccompanimentVerifier,
)
from app.services.note_context_model import ContextNoteModel, correction_keep
from app.services.note_context_model import residual_keep as runtime_keep

SEED = 260922
CEILING = .5
THRESHOLDS = (0., .0001, .0003, .001, .003, .01, .02, .03, .05, .075, .1, .15, .2)


def baseline_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [CHECKPOINT, *(path for path, _ in CONTEXT_CHECKPOINTS)]}


def prepare_items(directory, items, verifier):
    """Align only exact shared events; preserve all other deployed decisions."""
    prepared = []
    for item in items:
        x, events = previous_events(directory, item)
        lookup = {tuple(event): index for index, event in enumerate(item['events'])}
        indices = np.asarray([lookup.get(tuple(event), -1) for event in events], dtype=int)
        shared = indices >= 0
        context_x = np.zeros((len(events), 52), dtype=np.float32)
        context_x[shared] = item['x'][indices[shared]]
        # Training and production must see the same legacy acoustic features.
        np.testing.assert_array_equal(context_x[shared, :26], x[shared])
        with torch.inference_mode():
            p = torch.sigmoid(verifier.model(torch.from_numpy(
                (x - verifier.mean) / verifier.scale))).numpy()
        context = np.ones((2, len(events)))
        for index, head in enumerate(verifier.context_models):
            context[index, shared] = head.probability(context_x[shared])
        keep = correction_keep(p, context, shared, threshold=verifier.threshold,
                               prune=verifier.context_models[0].threshold)
        # Recompute one-to-one targets against actual deployed candidates.
        y, mask = targets(item['reference'], events)
        prepared.append({**item, 'x': context_x, 'events': events, 'p': p,
                         'shared': shared, 'keep': keep, 'y': y, 'mask': mask})
    return prepared


def residual_keep(item, probability, threshold):
    """A residual model may only reject shared, uncertain, retained events."""
    return runtime_keep(item['keep'], item['p'], item['shared'], probability,
                        threshold=threshold, ceiling=CEILING)


def evaluate(items, probabilities, threshold):
    rows = []
    for item, probability in zip(items, probabilities, strict=True):
        before = item['events'][item['keep']]
        after = item['events'][residual_keep(item, probability, threshold)]
        old, new = [metrics(item['reference'], events, item['seconds'])
                    for events in (before, after)]
        preserved = matched_references(item['reference'], before).issubset(
            matched_references(item['reference'], after))
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'deployed_v3': old,
                     'candidate': new, 'matched_references_preserved': preserved,
                     'passes': preserved and new['false_positives'] <= old['false_positives']})
    return {'threshold': threshold, 'passes': all(row['passes'] for row in rows),
            'false_notes_removed': sum(r['deployed_v3']['false_positives'] -
                                       r['candidate']['false_positives'] for r in rows),
            'failed_recordings': [r['id'] for r in rows if not r['passes']],
            'aggregate': {corpus: {system: aggregate([r[system] for r in rows if r['corpus'] == corpus])
                                  for system in ('deployed_v3', 'candidate')}
                          for corpus in sorted({r['corpus'] for r in rows})},
            'per_recording': rows}


def train(directory, output):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    write(output / 'label-audit.json', audit(training, validation))
    verifier = AccompanimentVerifier()
    training, validation = [prepare_items(directory, items, verifier) for items in (training, validation)]
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
            weights.append(np.where(y == 1, 6., 1.) / (len(records) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus, 'positive': int(y.sum()),
                           'negative': int(len(y) - y.sum())})
    x, y, weight = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(weights)
    weight /= weight.mean()
    settings = {'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 40,
                'learning_rate': .05, 'l2_regularization': 3.,
                'early_stopping': False, 'random_state': SEED}
    write(output / 'run.json', {'settings': settings, 'positive_weight': 6.,
        'training_clips': len(training), 'training_events': len(y),
        'validation_clips': len(validation), 'baseline_hashes': baseline_hashes(),
        'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5,
        'ceiling': CEILING, 'test_used_for_selection': False,
        'scope': 'Only cached licensed corpus features; no user recordings',
        'weighting': 'Equal corpora and recordings before positive-event weight',
        'training_counts': counts})
    model = HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight)
    probabilities = [model.predict_proba(item['x'])[:, 1] for item in validation]
    best, search = None, []
    for threshold in THRESHOLDS:
        result = evaluate(validation, probabilities, threshold)
        search.append({k: v for k, v in result.items() if k != 'per_recording'})
        if result['passes'] and result['false_notes_removed'] > 0 and (
                best is None or result['false_notes_removed'] > best['false_notes_removed']):
            best = result
    write(output / 'search.json', search)
    # Archive the trained model even if it is ineligible; never overwrite runs.
    chosen = best['threshold'] * .5 if best else 0.
    export(model, chosen, output / 'candidate.npz', validation)
    result = evaluate(validation, probabilities, chosen)
    write(output / 'validation.json', result)
    selection = {'selected': bool(best and result['passes'] and result['false_notes_removed'] > 0),
                 'checkpoint_sha256': hashlib.sha256((output / 'candidate.npz').read_bytes()).hexdigest(),
                 'threshold': chosen, 'ceiling': CEILING, 'baseline_hashes': baseline_hashes(),
                 'selection': 'Most false notes removed with no matched reference loss in any validation clip; halve threshold before testing',
                 'test_not_evaluated': True}
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


def cached_external(directory, path, audio_base):
    """Read existing dataset features only; never infer or render during regression."""
    for entry in json.loads(path.read_text())['items']:
        entry = {**entry, 'audio': str((audio_base / entry['audio']).resolve()),
                 'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True}
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        with np.load(directory / 'note-verifier-features' / (entry['id'] + '.npz'), allow_pickle=False) as saved:
            if str(saved['identity']) != digest:
                raise ValueError(f'Stale external feature cache: {entry["id"]}')
            fields = ('x', 'events', 'reference', 'seconds', 'legacy_events', 'legacy_x')
            yield {**entry, **{key: saved[key] for key in fields if key in saved}}


def test(directory, output, fresh=None):
    destination = output / ('fresh.json' if fresh else 'regression.json')
    if destination.exists():
        raise ValueError('Preserve the consumed evaluation; do not retune on it')
    selection = json.loads((output / 'selection.json').read_text())
    if not selection['selected']:
        raise ValueError('Candidate did not qualify on validation')
    if baseline_hashes() != selection['baseline_hashes']:
        raise ValueError('Deployed baseline changed after freeze')
    checkpoint = output / 'candidate.npz'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with np.load(checkpoint, allow_pickle=False) as saved:
        model = ContextNoteModel(saved)
    if model.threshold != selection['threshold'] or selection['ceiling'] != CEILING:
        raise ValueError('Candidate policy changed after freeze')
    if fresh:
        items = list(cached_external(directory, fresh, directory))
    else:
        items = load_data(directory, 'test')
        for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                     'later-vienna-context', 'oxford-later-context'):
            base = directory if name in ('later-vienna-context', 'oxford-later-context') else directory.parent
            items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    prepared = prepare_items(directory, items, AccompanimentVerifier())
    result = evaluate(prepared, [model.probability(item['x']) for item in prepared], model.threshold)
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  scope='Unscored dataset tails; same test performer' if fresh else 'Consumed regression data; not independent test accuracy')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='residual-verifier-v1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    destination = args.directory / args.run
    if args.test or args.fresh:
        test(args.directory, destination, args.fresh)
    else:
        train(args.directory, destination)
