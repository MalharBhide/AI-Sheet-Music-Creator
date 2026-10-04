"""Fit conservative consensus heads against pinned website V7, from licensed caches."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
import soundfile as sf
import torch
from app.services.left_baseline_evidence import (
    EVIDENCE_VERSION,
    RELATION_EVIDENCE_NAMES,
    acoustic_evidence,
    append_evidence,
)
from app.services.note_context_model import (
    ContextNoteModel,
    consensus_keep,
    shared_events,
)
from audit_verifier_labels import audit
from cache_note_verifier import targets
from note_relations import ALL_NAMES, EDGE_SECONDS, RELATION_VERSION, window_eligible
from sklearn.ensemble import HistGradientBoostingClassifier
from train_left_relations_v7 import hashes as v6_hashes
from train_left_relations_v7 import prepare as prepare_v6
from train_left_relations_v7 import score as score_v6
from train_note_verifier import load_data, write
from train_residual_verifier import THRESHOLDS, cached_external

ASSETS = Path(__file__).resolve().parents[1] / 'backend/app/assets'
V7_MODELS = (
    ('left-relations-v7.npz', '50c4a615d36a835459ca9f1d40879013e987dc3424c0f9496cff64b67353a681', .0375,
     {'feature_names': ALL_NAMES, 'feature_version': RELATION_VERSION}),
    ('left-guardian-v7.npz', '34d70e8e621f6809d7066cd5a80011299464280406f0a0ebafbf4ed0b76f0cda', .3, {}),
)
GUARDIAN_THRESHOLDS = (.1, .2, .4, .6)


def hashes():
    result = v6_hashes()
    for name, digest, _, _ in V7_MODELS:
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen website V7 baseline changed')
        result[name] = digest
    return result


def advance_baseline(item, relation, guardian):
    """Recompute labels after V7 removes duplicates; never restore older rejections."""
    item['keep'] = consensus_keep(item['keep'], item['p'], item['shared'],
        relation.probability(item['x']), guardian.probability(item['x'][:, :52]),
        threshold=relation.threshold, guardian_threshold=guardian.threshold)
    item['baseline_keep'] = item['keep'].copy()
    y, mask = targets(item['reference'], item['events'][item['keep']])
    item['y'], item['mask'] = np.zeros(len(item['events'])), np.zeros(len(item['events']), dtype=bool)
    item['y'][item['keep']], item['mask'][item['keep']] = y, mask
    return item


def prepare(directory, items, baseline_evidence=False, full_register=False):
    hashes()
    heads = []
    for name, _, threshold, contract in V7_MODELS:
        with np.load(ASSETS / name, allow_pickle=False) as saved:
            head = ContextNoteModel(saved, **contract)
        if head.threshold != threshold:
            raise ValueError('Baseline threshold changed')
        heads.append(head)
    prepared = [advance_baseline(item, *heads) for item in prepare_v6(directory, items)]
    if full_register:
        for raw, item in zip(items, prepared, strict=True):
            duration = min(30., sf.info(directory / item['audio']).duration)
            item['shared'] = shared_events(item['events'], raw['events']) & window_eligible(item['events'], duration)
    if baseline_evidence:
        for item in prepared:
            item['x'] = append_evidence(item['x'], heads[0].probability(item['x']),
                                        heads[1].probability(item['x'][:, :52]))
    return prepared


def score(items, probabilities, threshold, memo=None, *, ceiling=.5, preserve_coverage=False):
    result = score_v6(items, probabilities, threshold, memo, ceiling=ceiling,
                      preserve_coverage=preserve_coverage)
    # Copy cached rows instead of mutating the V6 helper's memoized objects.
    result['per_recording'] = [{**{k: v for k, v in row.items() if k != 'deployed_v6'},
                               'deployed_v7': row['deployed_v6']} for row in result['per_recording']]
    result['aggregate'] = {corpus: {'deployed_v7': values['deployed_v6'], 'candidate': values['candidate']}
                           for corpus, values in result['aggregate'].items()}
    return result


def acoustic_view(x):
    return x[:, :52] if x.ndim == 2 and x.shape[1] == 72 else acoustic_evidence(x)


def probabilities(models, x, threshold, guardian_threshold):
    """Use the same strict threshold comparisons as production, including ties."""
    relation = models[0].predict_proba(x)[:, 1]
    guardian = models[1].predict_proba(acoustic_view(x))[:, 1]
    return np.where((relation < threshold) & (guardian < guardian_threshold), 0., 1.)


def train(directory, output, extras, profile="guarded"):
    if profile not in ("guarded", "expanded", "baseline-evidence", "full-accompaniment"):
        raise ValueError("Unknown predeclared training profile")
    positive_weight = 10. if profile == "guarded" else 8.
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    identities = {item['id'] for item in training + validation}
    for path in extras:
        for item in cached_external(directory, path, directory):
            if item['group'] not in ('train', 'validation') or item['id'] in identities:
                raise ValueError('Test data or duplicate passages cannot enter fitting/selection')
            identities.add(item['id'])
            item['y'], item['mask'] = targets(item['reference'], item['events'])
            (training if item['group'] == 'train' else validation).append(item)
    write(output / 'label-audit.json', audit(training, validation))
    baseline_evidence = profile in ('baseline-evidence', 'full-accompaniment')
    full_register = profile == 'full-accompaniment'
    training, validation = [prepare(directory, items, baseline_evidence, full_register) for items in (training, validation)]
    features, labels, weights, counts = [], [], [], []
    for corpus in sorted({item['corpus'] for item in training}):
        items = [item for item in training if item['corpus'] == corpus]
        for item in items:
            mask = item['mask'] & item['keep'] & item['shared']
            x, y = item['x'][mask], item['y'][mask]
            if not len(y):
                continue
            features.append(x)
            labels.append(y)
            weights.append(np.where(y == 1, positive_weight, 1.) / (len(items) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus, 'positive': int(y.sum()), 'negative': int(len(y) - y.sum())})
    x, y, weight = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    weight /= weight.mean()
    settings = {'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 60,
                'learning_rate': .05, 'l2_regularization': 8., 'early_stopping': False, 'random_state': 261006}
    if profile != 'guarded':
        settings.update(min_samples_leaf=40, l2_regularization=4., random_state=261008)
    guardian_settings = {**settings, 'max_leaf_nodes': 15, 'random_state': 261007 if profile == 'guarded' else 261009}
    if baseline_evidence:
        settings['random_state'], guardian_settings['random_state'] = 261010, 261011
    if full_register:
        settings['random_state'], guardian_settings['random_state'] = 261012, 261013
    feature_names = RELATION_EVIDENCE_NAMES if baseline_evidence else ALL_NAMES
    feature_version = EVIDENCE_VERSION if baseline_evidence else RELATION_VERSION
    run = {'settings': settings, 'guardian_settings': guardian_settings, 'positive_weight': positive_weight, 'profile': profile,
           'training_clips': len(training), 'validation_clips': len(validation), 'training_events': len(y),
           'training_counts': counts, 'baseline_hashes': hashes(), 'threshold_grid': THRESHOLDS,
           'guardian_threshold_grid': GUARDIAN_THRESHOLDS, 'selection_safety_factor': .5,
           'feature_names': feature_names, 'feature_version': feature_version, 'edge_seconds': EDGE_SECONDS,
           'extra_manifest_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extras},
           'sklearn_version': sklearn.__version__, 'ceiling': .5, 'pitch_range': [36, 96] if full_register else [36, 60],
           'policy': 'v7-all-accompaniment-consensus' if full_register else 'v7-residual-consensus', 'test_used_for_selection': False,
           'scope': 'Cached licensed training data only; no user audio or score generation',
           'preservation': 'Every matched attack and hold, no per-recording false-note increase'}
    write(output / 'run.json', run)
    models = [HistGradientBoostingClassifier(**settings).fit(x, y, sample_weight=weight),
              HistGradientBoostingClassifier(**guardian_settings).fit(acoustic_view(x), y, sample_weight=weight)]
    context = [models[0].predict_proba(item['x'])[:, 1] for item in validation]
    acoustic = [models[1].predict_proba(acoustic_view(item['x']))[:, 1] for item in validation]
    best, search, memo = None, [], {}
    for guardian in GUARDIAN_THRESHOLDS:
        for threshold in THRESHOLDS:
            raw, margin = [score(validation, [np.where((c < at) & (a < g), 0., 1.)
                           for c, a in zip(context, acoustic, strict=True)], at, memo)
                           for at, g in ((threshold, guardian), (threshold * .5, guardian * .5))]
            search.append({'guardian_raw': guardian, 'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                           'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0 and (
                    best is None or margin['false_notes_removed'] > best[0]['false_notes_removed']):
                best = margin, guardian * .5
    result, guardian = best if best else (score(validation, [np.ones(len(item['events'])) for item in validation], 0.), 0.)
    with (output / 'candidate.pickle').open('wb') as stream:
        pickle.dump(models, stream)  # Locally fitted research artifact; never shipped to the website.
    selection = {'selected': best is not None, 'threshold': result['threshold'], 'guardian_threshold': guardian,
                 'checkpoint_sha256': hashlib.sha256((output / 'candidate.pickle').read_bytes()).hexdigest(),
                 'baseline_hashes': hashes(), 'policy': run['policy'], 'feature_version': feature_version,
                 'edge_seconds': EDGE_SECONDS, 'ceiling': .5, 'pitch_range': run['pitch_range'],
                 'selection': 'Validation-only joint grid; every recording passes at raw and half-margin thresholds',
                 'test_not_evaluated': True}
    write(output / 'search.json', search)
    write(output / 'validation.json', result)
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


def load_candidate(output):
    selection = json.loads((output / 'selection.json').read_text())
    run = json.loads((output / 'run.json').read_text())
    if (not selection['selected'] or selection['baseline_hashes'] != hashes()
            or selection['policy'] not in ('v7-residual-consensus', 'v7-all-accompaniment-consensus')
            or run['policy'] != selection['policy']
            or selection['feature_version'] != run['feature_version']
            or selection['feature_version'] not in (RELATION_VERSION, EVIDENCE_VERSION)
            or selection['edge_seconds'] != EDGE_SECONDS
            or selection['ceiling'] != .5
            or selection['pitch_range'] != ([36, 96] if selection['policy'] == 'v7-all-accompaniment-consensus' else [36, 60])
            or run['baseline_hashes'] != selection['baseline_hashes']
            or selection['threshold'] not in {value * .5 for value in run['threshold_grid']}
            or selection['guardian_threshold'] not in {value * .5 for value in run['guardian_threshold_grid']}
            or not 0 < selection['threshold'] <= 1 or not 0 < selection['guardian_threshold'] <= 1):
        raise ValueError('Ineligible candidate or changed frozen policy')
    checkpoint = output / 'candidate.pickle'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after freeze')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    widths = [74, 54] if selection['feature_version'] == EVIDENCE_VERSION else [72, 52]
    if [model.n_features_in_ for model in models] != widths:
        raise ValueError('Unexpected model feature contract')
    return selection, models


def regression_items(directory):
    items = load_data(directory, 'test')
    for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                 'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails',
                 'relational-piano-fresh', 'relational-piano-fresh-90'):
        base = directory.parent if name.startswith('note-verifier-') else directory
        items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    return items


def test(directory, output, fresh=None):
    destination = output / ('fresh.json' if fresh else 'regression.json')
    if destination.exists():
        raise ValueError('Preserve consumed evaluation; do not retune on regression')
    selection, models = load_candidate(output)
    if fresh:
        regression = output / 'regression.json'
        if not regression.exists() or not json.loads(regression.read_text())['passes']:
            raise ValueError('Preserve fresh evaluation until regression passes')
        items = list(cached_external(directory, fresh, directory))
        if not items or any(item['group'] != 'test' for item in items):
            raise ValueError('Fresh evaluation requires reserved test performers')
    else:
        items = regression_items(directory)
    torch.set_num_threads(2)
    items = prepare(directory, items, selection['feature_version'] == EVIDENCE_VERSION,
                    selection['policy'] == 'v7-all-accompaniment-consensus')
    result = score(items, [probabilities(models, item['x'], selection['threshold'], selection['guardian_threshold'])
                          for item in items], selection['threshold'])
    result.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  guardian_threshold=selection['guardian_threshold'],
                  selection_sha256=hashlib.sha256((output / 'selection.json').read_bytes()).hexdigest(),
                  scope='Later unscored passages, same reserved test performers/compositions' if fresh
                  else 'Consumed regression including all eight V7 later piano sections; not independent generalization')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='left-consensus-v8')
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--profile', choices=('guarded', 'expanded', 'baseline-evidence', 'full-accompaniment'), default='guarded')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    output = args.directory / args.run
    if args.test or args.fresh:
        test(args.directory, output, args.fresh)
    else:
        train(args.directory, output, args.extra, args.profile)
