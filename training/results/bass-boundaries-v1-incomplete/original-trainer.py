"""Train hold/repeat boundary heads against the pinned shipped V10 bass filter.

Only existing licensed/original caches enter fitting. Later checkpoints cannot
replace an earlier winner unless validation improves without losing any matched
attack, offset or pitch span. Frozen test failures cannot select a runner-up.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from bass_boundary_evidence import VERSION, acoustic_view, annotate, features, merge, pitch_union
from bass_predictions import probability
from bass_training_data import load
from current_bass_baseline import hashes, prepare
from evaluate_context_correction import matched_references
from pitch_interval_coverage import compare
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_bass_consensus import at_stage
from train_left_hand_verifier import held_matches
from train_note_verifier import aggregate, metrics

PROFILES = (
    {'name': 'guarded', 'max_iter': 350, 'positive_weight': 30., 'min_samples_leaf': 40,
     'max_leaf_nodes': 15, 'l2_regularization': 8.},
    {'name': 'recall', 'max_iter': 500, 'positive_weight': 60., 'min_samples_leaf': 20,
     'max_leaf_nodes': 15, 'l2_regularization': 8.},
)
STAGES = (100, 200, 350, 500)
THRESHOLDS = (.001, .002, .005, .01, .025, .05, .1)
GUARDIANS = (.05, .1, .2)


def contracts():
    root = Path(__file__).resolve().parents[1]
    return {name: digest(root / name) for name in (
        'training/train_bass_boundaries.py', 'training/bass_boundary_evidence.py',
        'training/bass_boundary_release.py',
        'training/current_bass_baseline.py', 'training/repeat_boundary_features.py',
        'training/bass_predictions.py', 'training/bass_training_data.py',
        'training/train_bass_consensus.py', 'training/pitch_interval_coverage.py',
        'training/evaluate_context_correction.py', 'training/train_left_hand_verifier.py',
        'training/train_note_verifier.py')}


def collect(roots):
    cohorts, identities = [[], []], set()
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        if (not manifest['no_test_inference'] or manifest['plan_sha256'] != digest(root / 'plan.json')
                or any(item['group'] not in ('train', 'validation') for item in manifest['items'])):
            raise ValueError('Test or stale preparations cannot enter boundary fitting')
        for group, destination in zip(('train', 'validation'), cohorts, strict=True):
            for item in load(root, group):
                if item['id'] in identities or item['id'].startswith('bass-held-v1-'):
                    raise ValueError('Duplicate or consumed regression in fitting')
                identities.add(item['id'])
                destination.append(item)
    if {item['source_group'] for item in cohorts[0]} & {item['source_group'] for item in cohorts[1]}:
        raise ValueError('Boundary source groups leak across fitting and selection')
    return [annotate(prepare(items)) for items in cohorts]


def audit(items):
    rows = [{'id': item['id'], 'corpus': item['corpus'], 'group': item['group'],
             'retained_notes': int(item['keep'].sum()), 'pairs': len(item['pairs']),
             'repeats': int(item['boundary_y'][item['boundary_mask']].sum()),
             'split_holds': int(np.sum(item['boundary_mask'] & (item['boundary_y'] == 0))),
             'ambiguous': int(np.sum(~item['boundary_mask']))} for item in items]
    return {'items': rows, 'counts': {group: {key: sum(row[key] for row in rows if row['group'] == group)
            for key in ('retained_notes', 'pairs', 'repeats', 'split_holds', 'ambiguous')}
            for group in ('train', 'validation')}}


def matrix(items, positive_weight):
    xs, ys, weights, counts = [], [], [], []
    for corpus in sorted({item['corpus'] for item in items}):
        records = [item for item in items if item['corpus'] == corpus]
        for item in records:
            mask = item['boundary_mask']
            y = item['boundary_y'][mask]
            if not len(y):
                continue
            xs.append(features(item, item['pairs'])[mask])
            ys.append(y)
            weights.append(np.where(y == 1, positive_weight, 1.) / (len(records) * len(y)))
            counts.append({'id': item['id'], 'corpus': corpus, 'repeats': int(y.sum()),
                           'split_holds': int(len(y) - y.sum())})
    if not ys or len(np.unique(np.concatenate(ys))) != 2:
        raise ValueError('Boundary supervision must include real repeats and split holds')
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    return x, y, weight / weight.mean(), counts


def predictions(models, items):
    return [(probability(models[0], features(item, item['pairs'])),
             probability(models[1], features(item, item['pairs'], acoustic=True))) for item in items]


def score(items, probabilities, threshold, guardian, memo=None):
    rows = []
    for item, (p, g) in zip(items, probabilities, strict=True):
        before = item['events'][item['baseline_keep']]
        after, merged = merge(item, item['pairs'], p, g, threshold, guardian)
        key = (id(item), after.tobytes())
        if memo is not None and key in memo:
            rows.append(memo[key])
            continue
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        coverage = compare(item['pitch_reference'], before, after)
        union = pitch_union(before) == pitch_union(after)
        row = {'id': item['id'], 'corpus': item['corpus'], 'deployed_v10': old, 'candidate': new,
               'merged_boundaries': len(merged), 'matched_attacks_preserved': attacks,
               'matched_offsets_preserved': holds, 'pitch_support_preserved': coverage['passes'],
               'entire_pitch_time_union_preserved': union,
               'passes': attacks and holds and coverage['passes'] and union
               and new['false_positives'] <= old['false_positives']}
        rows.append(row)
        if memo is not None:
            memo[key] = row
    return {'threshold': threshold, 'guardian_threshold': guardian,
            'passes': bool(rows) and all(row['passes'] for row in rows),
            'false_notes_removed': sum(row['deployed_v10']['false_positives'] - row['candidate']['false_positives'] for row in rows),
            'merged_boundaries': sum(row['merged_boundaries'] for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']],
            'aggregate': {key: aggregate([row[key] for row in rows]) for key in ('deployed_v10', 'candidate')},
            'per_recording': rows}


def select(items, probabilities, memo):
    best, rows = None, []
    for guardian in GUARDIANS:
        for threshold in THRESHOLDS:
            raw = score(items, probabilities, threshold, guardian, memo)
            margin = score(items, probabilities, threshold * .5, guardian * .5, memo)
            rows.append({'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                         'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if (raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0
                    and (best is None or margin['false_notes_removed'] > best['false_notes_removed'])):
                best = margin
    return best, rows


def train(roots, output, regressions):
    output.mkdir(exist_ok=False)
    fitted_sources = {item['source_group'] for root in roots
                      for item in json.loads((root / 'manifest.json').read_text())['items']}
    test_ids = set()
    for root in regressions:
        manifest = json.loads((root / 'manifest.json').read_text())
        if not manifest['items'] or any(item['group'] != 'test' or item['source_group'] in fitted_sources
                                        for item in manifest['items']):
            raise ValueError('Consumed regression partitions overlap fitting or are not all test')
        for item in manifest['items']:
            if item['id'] in test_ids:
                raise ValueError('Duplicate consumed regression clip')
            test_ids.add(item['id'])
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'code_sha256': contracts(),
            'manifests': {str(root): digest(root / 'manifest.json') for root in roots},
            'consumed_regression_manifests': {str(root): digest(root / 'manifest.json') for root in regressions},
            'profiles': list(PROFILES), 'stages': list(STAGES), 'threshold_grid': list(THRESHOLDS),
            'guardian_grid': list(GUARDIANS), 'safety_factor': .5, 'feature_counts': [112, 60],
            'sklearn_version': sklearn.__version__,
            'selection': 'Validation only: maximum safe false attack reduction, then lower log loss, fewer trees and fixed profile order.',
            'gates': 'Every matched attack and offset, each reference pitch span, entire detected pitch-time union, per-recording false-note nonincrease.',
            'scope': 'Offline balanced bass only; no user audio, transcription jobs or score generation. No release without independent gates.'}
    preserve(output / 'plan.json', plan)
    training, validation = collect(roots)
    preserve(output / 'supervision.json', audit(training + validation))
    vx, vy, vw, _ = matrix(validation, 1.)
    selections = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        config = {k: v for k, v in profile.items() if k not in ('name', 'positive_weight')}
        config.update(learning_rate=.04, early_stopping=False, random_state=261141 + index * 2)
        guardian_config = {**config, 'max_leaf_nodes': 7, 'random_state': 261142 + index * 2}
        preserve(destination / 'run.json', {'settings': config, 'guardian_settings': guardian_config,
                 'training_counts': counts, 'training_clips': len(training), 'validation_clips': len(validation),
                 'training_events': len(y), 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, curves = [], {}
        for name, settings, tx, test_x in [('repeat', config, x, vx),
                                         ('guardian', guardian_config, acoustic_view(x), acoustic_view(vx))]:
            print(f'Fitting bass boundary {profile["name"]} {name}: {len(y)} pairs', flush=True)
            model = HistGradientBoostingClassifier(**settings).fit(tx, y, sample_weight=weight)
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, prediction[:, 1],
                            sample_weight=vw, labels=[0., 1.]))}
                            for step, prediction in enumerate(model.staged_predict_proba(test_x), 1) if step in STAGES]
            models.append(model)
        preserve(destination / 'learning-curves.json', curves)
        best, stages, memo = None, [], {}
        for trees in [step for step in STAGES if step <= profile['max_iter']]:
            staged = at_stage(models, trees)
            chosen, search = select(validation, predictions(staged, validation), memo)
            loss = sum(next(row['validation_log_loss'] for row in curves[name] if row['trees'] == trees)
                       for name in ('repeat', 'guardian')) / 2
            row = {'trees': trees, 'validation_log_loss': loss, 'selected': chosen is not None,
                   'false_notes_removed': chosen['false_notes_removed'] if chosen else 0}
            stages.append(row)
            preserve(destination / f'search-{trees}.json', search)
            if chosen:
                key = (chosen['false_notes_removed'], -loss, -trees)
                if best is None or key > best[0]:
                    best = key, staged, chosen, trees, loss
            print(json.dumps({'profile': profile['name'], **row}), flush=True)
        preserve(destination / 'stages.json', stages)
        with (destination / 'candidate.pickle').open('wb') as stream:
            pickle.dump(best[1] if best else models, stream)
        if best:
            preserve(destination / 'validation.json', best[2])
        selection = {'name': profile['name'], 'selected': best is not None,
                     'validation_false_notes_removed': best[2]['false_notes_removed'] if best else 0,
                     'threshold': best[2]['threshold'] if best else 0.,
                     'guardian_threshold': best[2]['guardian_threshold'] if best else 0.,
                     'trees': best[3] if best else profile['max_iter'],
                     'validation_log_loss': best[4] if best else None,
                     'checkpoint_sha256': digest(destination / 'candidate.pickle'),
                     'validation_sha256': digest(destination / 'validation.json') if best else None,
                     'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False}
        preserve(destination / 'selection.json', selection)
        selections.append(selection)
    eligible = [row for row in selections if row['selected']]
    winner = max(eligible, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['trees'])) if eligible else None
    preserve(output / 'batch-selection.json', {'selected': winner is not None, 'winner': winner,
             'candidates': selections, 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
    print(json.dumps({'batch_winner': winner}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--regression', type=Path, action='append', default=[])
    args = parser.parse_args()
    train(args.roots, args.output, args.regression)
