"""Fit bass-only heads, selecting gated validation checkpoints before any test.

The production bass baseline is the bounded Basic Pitch decoder, not V9's
accompaniment classifier. Each matched attack, offset and supplied pitch interval
must be retained in every recording. No threshold/model tuning uses test data.
"""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from app.services.note_evidence import CONTEXT_VERSION
from bass_training_data import EDGE_SECONDS, NAMES, VERSION, load
from evaluate_context_correction import matched_references
from pitch_interval_coverage import compare
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_left_hand_verifier import held_matches
from train_note_verifier import aggregate, metrics

POLICY = 'balanced-bass-consensus-v1'
THRESHOLDS = (.01, .02, .05, .1, .2, .4, .6)
GUARDIANS = (.05, .1, .2, .4, .6)
STAGES = (100, 200, 400, 500, 600)
PROFILES = (
    {'name': 'regularized', 'max_iter': 400, 'max_leaf_nodes': 15,
     'min_samples_leaf': 40, 'l2_regularization': 8., 'positive_weight': 16.},
    {'name': 'capacity', 'max_iter': 600, 'max_leaf_nodes': 31,
     'min_samples_leaf': 20, 'l2_regularization': 4., 'positive_weight': 10.},
    {'name': 'recall', 'max_iter': 500, 'max_leaf_nodes': 15,
     'min_samples_leaf': 30, 'l2_regularization': 8., 'positive_weight': 40.},
    {'name': 'shallow', 'max_iter': 500, 'max_leaf_nodes': 7,
     'min_samples_leaf': 60, 'l2_regularization': 8., 'positive_weight': 20.},
)


def matrix(items, positive_weight):
    x, y, weights, counts = [], [], [], []
    for corpus in sorted({item['corpus'] for item in items}):
        records = [item for item in items if item['corpus'] == corpus]
        for item in records:
            mask = item['mask'] & item['eligible']
            labels = item['y'][mask]
            if not len(labels):
                continue
            x.append(item['x'][mask])
            y.append(labels)
            weights.append(np.where(labels == 1, positive_weight, 1.) / (len(records) * len(labels)))
            counts.append({'id': item['id'], 'corpus': corpus,
                           'positive': int(labels.sum()), 'negative': int(len(labels) - labels.sum())})
    if not x:
        raise ValueError('No eligible labeled bass candidates')
    x, y, weight = np.concatenate(x), np.concatenate(y), np.concatenate(weights)
    if len(np.unique(y)) != 2:
        raise ValueError('Bass fitting/validation requires both positive and negative supervision')
    return x, y, weight / weight.mean(), counts


def score(items, context, acoustic, threshold, guardian, memo=None):
    rows = []
    for item, p, g in zip(items, context, acoustic, strict=True):
        p, g = np.asarray(p), np.asarray(g)
        count = len(item['events'])
        if (p.shape != (count,) or g.shape != (count,) or not np.isfinite(p).all()
                or not np.isfinite(g).all() or np.any((p < 0) | (p > 1))
                or np.any((g < 0) | (g > 1)) or not 0 <= threshold <= 1 or not 0 <= guardian <= 1):
            raise ValueError('Invalid paired bass confidence')
        keep = ~(item['eligible'] & (p < threshold) & (g < guardian))
        key = (id(item), keep.tobytes())
        if memo is not None and key in memo:
            rows.append(memo[key])
            continue
        before, after = item['events'], item['events'][keep]
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        coverage = compare(item['pitch_reference'], before, after)
        row = {'id': item['id'], 'corpus': item['corpus'], 'baseline_bass': old, 'candidate': new,
               'matched_references_preserved': attacks, 'held_references_preserved': holds,
               'reference_pitch_coverage_preserved': coverage['passes'],
               'lost_reference_pitch_seconds': coverage['lost_reference_seconds'],
               'passes': attacks and holds and coverage['passes']
               and new['false_positives'] <= old['false_positives']}
        rows.append(row)
        if memo is not None:
            memo[key] = row
    return {'threshold': threshold, 'guardian_threshold': guardian,
            'passes': bool(rows) and all(row['passes'] for row in rows),
            'false_notes_removed': sum(row['baseline_bass']['false_positives'] - row['candidate']['false_positives'] for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']],
            'aggregate': {key: aggregate([row[key] for row in rows]) for key in ('baseline_bass', 'candidate')},
            'per_recording': rows}


def select(items, context, acoustic, memo):
    best, search = None, []
    for guardian in GUARDIANS:
        for threshold in THRESHOLDS:
            raw = score(items, context, acoustic, threshold, guardian, memo)
            margin = score(items, context, acoustic, threshold * .5, guardian * .5, memo)
            search.append({'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                           'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if (raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0
                    and (best is None or margin['false_notes_removed'] > best['false_notes_removed'])):
                best = margin
    return best, search


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = ('training/train_bass_consensus.py', 'training/bass_training_data.py',
             'training/bass_release.py', 'training/prepare_bass_regression.py',
             'training/prepare_slakh_bass_test.py',
             'training/export_bass_consensus.py', 'training/export_context_verifier.py',
             'training/prepare_slakh_bass.py', 'training/slakh_references.py',
             'training/pitch_support_targets.py', 'training/pitch_interval_coverage.py',
             'backend/app/services/note_evidence.py', 'backend/app/services/piano_transcription.py',
             'backend/app/services/source_separation.py')
    return {name: digest(root / name) for name in names}


def at_stage(models, trees):
    if trees not in STAGES or any(trees > len(model._predictors) for model in models):
        raise ValueError('Checkpoint depth is outside the frozen training stages')
    checkpoint = [copy.deepcopy(model) for model in models]
    for model in checkpoint:
        model._predictors = model._predictors[:trees]
    return checkpoint


def train(directory, output):
    output.mkdir(exist_ok=False)
    manifest = json.loads((directory / 'manifest.json').read_text())
    preparation = json.loads((directory / 'plan.json').read_text())
    if (manifest['plan_sha256'] != digest(directory / 'plan.json')
            or preparation['candidate_decoder'] != VERSION
            or not manifest['no_test_inference'] or any(i['group'] not in ('train', 'validation') for i in manifest['items'])):
        raise ValueError('Fitting/selection cannot consume test data or stale preparation')
    plan = {'policy': POLICY, 'profiles': list(PROFILES), 'stages': list(STAGES),
            'threshold_grid': list(THRESHOLDS), 'guardian_grid': list(GUARDIANS), 'margin': .5,
            'manifest_sha256': digest(directory / 'manifest.json'), 'preparation_sha256': digest(directory / 'plan.json'),
            'code_sha256': contracts(), 'feature_names': list(NAMES), 'feature_version': CONTEXT_VERSION,
            'sklearn_version': sklearn.__version__, 'edge_seconds': EDGE_SECONDS,
            'selection': 'Validation-only maximum gated false-note reduction; ties use lower validation loss, fewer trees, then fixed profile order.',
            'gates': 'Every matched attack, offset and supplied pitch interval preserved; no per-recording false-note increase at raw and half-margin.',
            'scope': 'Balanced bass only; no accompaniment/vocal/solo-piano routing changes; no user uploads or scores.'}
    preserve(output / 'plan.json', plan)
    training, validation = load(directory, 'train'), load(directory, 'validation')
    train_groups = {i['source_group'] for i in training}
    val_groups = {i['source_group'] for i in validation}
    if train_groups & val_groups or len(train_groups) != 12 or len(val_groups) != 4:
        raise ValueError('Source groups leaked across bass partitions')
    preserve(output / 'baseline-audit.json', {'items': [
        {'id': item['id'], 'group': item['group'], 'corpus': item['corpus'],
         'baseline_bass': metrics(item['reference'], item['events'], item['seconds']),
         'eligible': int(item['eligible'].sum()), 'supervised': int(np.sum(item['mask'] & item['eligible'])),
         'excluded_ambiguous': int(np.sum(~item['mask'] & item['eligible']))}
        for item in training + validation],
        'scope': 'Fixed training/validation only; clock and label diagnostics, not independent accuracy.'})
    vx, vy, vw, _ = matrix(validation, 1.)
    rows = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        settings = {key: value for key, value in profile.items() if key not in ('name', 'positive_weight')}
        settings.update(learning_rate=.04, early_stopping=False, random_state=261041 + index * 2)
        guardian_settings = {**settings, 'max_leaf_nodes': 7, 'random_state': 261042 + index * 2}
        preserve(destination / 'run.json', {'settings': settings, 'guardian_settings': guardian_settings,
                 'positive_weight': profile['positive_weight'], 'training_counts': counts,
                 'training_clips': len(training), 'validation_clips': len(validation), 'training_events': len(y),
                 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, staged, curves = [], [], {}
        for name, config, tx, test_x in [('context', settings, x, vx),
                                       ('guardian', guardian_settings, x[:, :26], vx[:, :26])]:
            print(f'Fitting bass {profile["name"]} {name}: {len(y)} events, {config["max_iter"]} trees', flush=True)
            model = HistGradientBoostingClassifier(**config).fit(tx, y, sample_weight=weight)
            predictions = {step: probability[:, 1] for step, probability in enumerate(model.staged_predict_proba(test_x), 1)
                           if step in STAGES or step == config['max_iter']}
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, prediction, sample_weight=vw, labels=[0., 1.]))}
                            for step, prediction in predictions.items()]
            models.append(model)
            staged.append(predictions)
        preserve(destination / 'learning-curves.json', curves)
        # Predict validation candidates at each declared checkpoint, retaining
        # the earliest protected checkpoint if later training worsens quality.
        best, stage_rows, memo = None, [], {}
        for trees in staged[0]:
            checkpoint = at_stage(models, trees)
            context = [checkpoint[0].predict_proba(item['x'])[:, 1] for item in validation]
            acoustic = [checkpoint[1].predict_proba(item['x'][:, :26])[:, 1] for item in validation]
            chosen, search = select(validation, context, acoustic, memo)
            loss = sum(next(row['validation_log_loss'] for row in curves[name] if row['trees'] == trees)
                       for name in ('context', 'guardian')) / 2
            stage_rows.append({'trees': trees, 'validation_log_loss': loss, 'selected': chosen is not None,
                               'false_notes_removed': chosen['false_notes_removed'] if chosen else 0})
            preserve(destination / f'search-{trees}.json', search)
            if chosen:
                key = (chosen['false_notes_removed'], -loss, -trees)
                if best is None or key > best[0]:
                    best = key, checkpoint, chosen, trees, loss
            print(json.dumps({'profile': profile['name'], **stage_rows[-1]}), flush=True)
        preserve(destination / 'stages.json', stage_rows)
        selected = best is not None
        chosen_models = best[1] if selected else models
        with (destination / 'candidate.pickle').open('wb') as stream:
            pickle.dump(chosen_models, stream)
        row = {'name': profile['name'], 'selected': selected, 'policy': POLICY,
               'validation_false_notes_removed': best[2]['false_notes_removed'] if selected else 0,
               'threshold': best[2]['threshold'] if selected else 0.,
               'guardian_threshold': best[2]['guardian_threshold'] if selected else 0.,
               'trees': best[3] if selected else profile['max_iter'], 'validation_log_loss': best[4] if selected else None,
               'plan_sha256': digest(output / 'plan.json'), 'checkpoint_sha256': digest(destination / 'candidate.pickle'),
               'test_not_evaluated': True}
        if selected:
            preserve(destination / 'validation.json', best[2])
        preserve(destination / 'selection.json', row)
        rows.append(row)
    eligible_rows = [row for row in rows if row['selected']]
    winner = max(eligible_rows, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['trees'])) if eligible_rows else None
    preserve(output / 'batch-selection.json', {'selected': winner is not None, 'winner': winner,
             'candidates': rows, 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
    print(json.dumps({'batch_winner': winner}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='bass-consensus-v1')
    args = parser.parse_args()
    train(args.directory, args.directory / args.run)
