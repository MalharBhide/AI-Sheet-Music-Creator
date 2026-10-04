"""Learn genuine repeats versus split holds against the pinned website V8.

This offline experiment uses only licensed corpus caches. Continuous pitch-time
coverage is preserved by construction and gated alongside reference attacks and
offsets. A validation winner is frozen before a single regression evaluation.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
import torch
from current_accompaniment_baseline import hashes, prepare
from evaluate_context_correction import matched_references
from pitch_interval_coverage import compare
from repeat_boundary_features import VERSION, features, merge, pitch_union
from repeat_boundary_supervision import boundaries, labels
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_broad_consensus import collect, digest
from train_left_consensus_v8 import regression_items
from train_left_hand_verifier import held_matches
from train_note_verifier import metrics, write

PROFILES = (
    {'name': 'guarded', 'max_iter': 350, 'positive_weight': 30., 'min_samples_leaf': 60,
     'max_leaf_nodes': 15, 'l2_regularization': 8.},
    {'name': 'recall', 'max_iter': 500, 'positive_weight': 50., 'min_samples_leaf': 40,
     'max_leaf_nodes': 15, 'l2_regularization': 8.},
)
THRESHOLDS = (.001, .002, .005, .01, .025, .05, .1)
GUARDIANS = (.05, .1, .2)


def annotate(items):
    for item in items:
        item['pairs'] = boundaries(item)
        item['boundary_y'], item['boundary_mask'] = labels(item['reference'], item['events'], item['pairs'])
    return items


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
            counts.append({'id': item['id'], 'corpus': corpus, 'genuine_repeats': int(y.sum()),
                           'split_holds': int(len(y) - y.sum())})
    if not ys:
        raise ValueError('No supervised boundaries')
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    return x, y, weight / weight.mean(), counts


def acoustic_view(x):
    # Paired 74-dimensional relation vectors contain acoustic columns 0:52
    # and pinned confidence columns 72:74; the last four columns are geometry.
    return np.column_stack((x[:, :52], x[:, 72:74], x[:, 74:126], x[:, 146:148], x[:, -4:]))


def predict(models, items):
    return [(models[0].predict_proba(features(item, item['pairs']))[:, 1],
             models[1].predict_proba(features(item, item['pairs'], acoustic=True))[:, 1])
            if len(item['pairs']) else (np.zeros(0), np.zeros(0)) for item in items]


def score(items, probabilities, threshold, guardian_threshold, memo=None):
    rows = []
    for item, (repeat, guardian) in zip(items, probabilities, strict=True):
        before = item['events'][item['baseline_keep']]
        after, merged = merge(item, item['pairs'], repeat, guardian, threshold, guardian_threshold)
        key = (id(item), after.tobytes())
        if memo is not None and key in memo:
            rows.append(memo[key])
            continue
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        coverage, union = compare(item['reference'], before, after), pitch_union(before) == pitch_union(after)
        row = {'id': item['id'], 'corpus': item['corpus'], 'deployed_v8': old, 'candidate': new,
               'merged_boundaries': len(merged), 'matched_references_preserved': attacks,
               'held_references_preserved': holds, 'reference_pitch_coverage_preserved': coverage['passes'],
               'pitch_time_union_preserved': union, 'lost_reference_pitch_seconds': coverage['lost_reference_seconds'],
               'passes': attacks and holds and coverage['passes'] and union
               and new['false_positives'] <= old['false_positives']}
        rows.append(row)
        if memo is not None:
            memo[key] = row
    return {'threshold': threshold, 'guardian_threshold': guardian_threshold,
            'passes': all(row['passes'] for row in rows),
            'false_notes_removed': sum(row['deployed_v8']['false_positives'] - row['candidate']['false_positives']
                                       for row in rows), 'merged_boundaries': sum(row['merged_boundaries'] for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']], 'per_recording': rows}


def select(items, probabilities):
    best, rows, memo = None, [], {}
    for guardian in GUARDIANS:
        for threshold in THRESHOLDS:
            raw = score(items, probabilities, threshold, guardian, memo)
            margin = score(items, probabilities, threshold * .5, guardian * .5, memo)
            rows.append({'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                         'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0 and (
                    best is None or margin['false_notes_removed'] > best['false_notes_removed']):
                best = margin
    return best, rows


def train(directory, output, extras):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'profiles': list(PROFILES),
            'threshold_grid': THRESHOLDS, 'guardian_grid': GUARDIANS, 'safety_factor': .5,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                            ('train_repeat_boundaries.py', 'repeat_boundary_features.py',
                             'repeat_boundary_supervision.py', 'current_accompaniment_baseline.py')},
            'extra_manifest_sha256': {str(path): digest(path) for path in extras},
            'sklearn_version': sklearn.__version__,
            'selection': 'One winner on gated validation gain, fixed plan-order tie break; no test tuning',
            'preservation': 'Per-recording attacks, offsets, annotated coverage and exact all-pitch interval union',
            'scope': 'Licensed cached corpus only; no uploads, transcription or scores; offline boundary model'}
    write(output / 'plan.json', plan)
    training, validation = collect(directory, extras)
    print('Preparing pinned V8 boundary evidence', flush=True)
    training, validation = [annotate(prepare(directory, items)) for items in (training, validation)]
    vx, vy, vw, _ = matrix(validation, 1.)
    selections = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        config = {k: v for k, v in profile.items() if k not in ('name', 'positive_weight')}
        config.update(learning_rate=.04, early_stopping=False, random_state=261030 + index * 2)
        guardian_config = {**config, 'random_state': 261031 + index * 2}
        write(destination / 'run.json', {'settings': config, 'guardian_settings': guardian_config,
              'positive_weight': profile['positive_weight'], 'training_counts': counts,
              'training_clips': len(training), 'validation_clips': len(validation), 'training_events': len(y),
              'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, curves = [], {}
        for name, settings, tx, test_x in [('repeat', config, x, vx),
                                         ('guardian', guardian_config, acoustic_view(x), acoustic_view(vx))]:
            print(f'Fitting {profile["name"]} {name}: {len(y)} boundaries', flush=True)
            model = HistGradientBoostingClassifier(**settings).fit(tx, y, sample_weight=weight)
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, prediction,
                            sample_weight=vw, labels=[0., 1.]))}
                            for step, prediction in enumerate(model.staged_predict_proba(test_x), 1)
                            if step % 50 == 0 or step == settings['max_iter']]
            models.append(model)
            write(destination / 'learning-curves.json', curves)
        result, search = select(validation, predict(models, validation))
        with (destination / 'candidate.pickle').open('wb') as stream:
            pickle.dump(models, stream)
        selection = {'name': profile['name'], 'selected': result is not None,
                     'validation_false_notes_removed': result['false_notes_removed'] if result else 0,
                     'threshold': result['threshold'] if result else 0.,
                     'guardian_threshold': result['guardian_threshold'] if result else 0.,
                     'checkpoint_sha256': digest(destination / 'candidate.pickle'),
                     'plan_sha256': digest(output / 'plan.json'), 'baseline_hashes': hashes()}
        write(destination / 'selection.json', selection)
        write(destination / 'search.json', search)
        if result:
            write(destination / 'validation.json', result)
        selections.append(selection)
        print(json.dumps(selection), flush=True)
    eligible = [row for row in selections if row['selected']]
    winner = max(eligible, key=lambda row: row['validation_false_notes_removed']) if eligible else None
    write(output / 'batch-selection.json', {'selected': winner is not None, 'winner': winner,
          'candidates': selections, 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
    print(json.dumps({'batch_winner': winner}, indent=2), flush=True)


def test(directory, output):
    destination = output / 'regression.json'
    if destination.exists():
        raise ValueError('Preserve consumed evaluation')
    batch = json.loads((output / 'batch-selection.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    eligible = [row for row in batch['candidates'] if row['selected']]
    if not eligible or not batch['selected']:
        raise ValueError('No eligible validation winner')
    winner = max(eligible, key=lambda row: row['validation_false_notes_removed'])
    if (batch['winner'] != winner or batch['plan_sha256'] != digest(output / 'plan.json')
            or plan['baseline_hashes'] != hashes() or winner['baseline_hashes'] != hashes()
            or winner['plan_sha256'] != digest(output / 'plan.json') or plan['profiles'] != list(PROFILES)
            or plan['threshold_grid'] != list(THRESHOLDS) or plan['guardian_grid'] != list(GUARDIANS)
            or plan['safety_factor'] != .5 or plan['version'] != VERSION
            or winner['threshold'] not in {v * .5 for v in THRESHOLDS}
            or winner['guardian_threshold'] not in {v * .5 for v in GUARDIANS}):
        raise ValueError('Changed frozen boundary selection')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Changed frozen boundary code')
    for filename, expected in plan['extra_manifest_sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Changed fitting manifest')
    source = output / winner['name']
    if json.loads((source / 'selection.json').read_text()) != winner:
        raise ValueError('Changed candidate selection')
    if digest(source / 'candidate.pickle') != winner['checkpoint_sha256']:
        raise ValueError('Changed candidate weights')
    with (source / 'candidate.pickle').open('rb') as stream:
        models = pickle.load(stream)
    if [m.n_features_in_ for m in models] != [152, 112]:
        raise ValueError('Changed boundary features')
    torch.set_num_threads(2)
    items = annotate(prepare(directory, regression_items(directory)))
    result = score(items, predict(models, items), winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(output / 'batch-selection.json'),
                  scope='Consumed regression only; no threshold retuning; no website release')
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='repeat-boundaries-v1')
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        test(args.directory, args.directory / args.run)
    else:
        train(args.directory, args.directory / args.run, args.extra)
