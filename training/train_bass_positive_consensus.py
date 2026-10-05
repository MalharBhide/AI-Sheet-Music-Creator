"""Fit a broader bass consensus with original timbres and licensed guitar.

The production bass baseline is the bounded Basic Pitch decoder, not V9's
accompaniment classifier. Each matched attack, offset and supplied pitch interval
must be retained in every recording. No threshold/model tuning uses test data.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from app.services.note_evidence import CONTEXT_VERSION
from bass_predictions import probability
from bass_training_data import EDGE_SECONDS, NAMES, VERSION, load
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_bass_consensus import (
    GUARDIANS,
    PROFILES,
    STAGES,
    THRESHOLDS,
    at_stage,
    matrix,
    select,
)
from train_bass_consensus import score as score
from train_note_verifier import metrics

POLICY = 'balanced-bass-positive-consensus-v2'


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = ('training/train_bass_positive_consensus.py', 'training/train_bass_consensus.py', 'training/bass_training_data.py',
             'training/bass_predictions.py',
             'training/bass_positive_release.py', 'training/prepare_bass_positive_data.py',
             'training/prepare_slakh_bass_positive_test.py',
             'training/export_bass_positive_consensus.py', 'training/export_context_verifier.py',
             'training/prepare_slakh_bass.py', 'training/slakh_references.py',
             'training/pitch_support_targets.py', 'training/pitch_interval_coverage.py',
             'backend/app/services/note_evidence.py', 'backend/app/services/piano_transcription.py',
             'backend/app/services/source_separation.py')
    return {name: digest(root / name) for name in names}


def extend(training, validation, extras):
    identities = {item['id'] for item in training + validation}
    if len(identities) != len(training) + len(validation):
        raise ValueError('Duplicate base bass clips')
    for extra in extras:
        other = json.loads((extra / 'manifest.json').read_text())
        if (not other['no_test_inference'] or other['plan_sha256'] != digest(extra / 'plan.json')
                or any(item['group'] not in ('train', 'validation') for item in other['items'])):
            raise ValueError('Reserved/test data or stale preparation cannot enter additional bass fitting')
        for group, destination in [('train', training), ('validation', validation)]:
            for item in load(extra, group):
                if item['id'] in identities or item['id'].startswith('bass-held-v1-'):
                    raise ValueError('Duplicate or consumed-regression examples cannot enter fitting')
                identities.add(item['id'])
                destination.append(item)
    if {item['source_group'] for item in training} & {item['source_group'] for item in validation}:
        raise ValueError('Additional bass source groups leak across selection')
    return training, validation


def train(directory, output, extras):
    output.mkdir(exist_ok=False)
    manifest = json.loads((directory / 'manifest.json').read_text())
    preparation = json.loads((directory / 'plan.json').read_text())
    if (manifest['plan_sha256'] != digest(directory / 'plan.json')
            or preparation['candidate_decoder'] != VERSION
            or not manifest['no_test_inference'] or any(i['group'] not in ('train', 'validation') for i in manifest['items'])):
        raise ValueError('Fitting/selection cannot consume test data or stale preparation')
    plan = {'policy': POLICY, 'profiles': list(PROFILES), 'stages': list(STAGES),
            'threshold_grid': list(THRESHOLDS), 'guardian_grid': list(GUARDIANS), 'margin': .5,
            'manifest_sha256': digest(directory / 'manifest.json'),
            'additional_manifests': {str(extra): digest(extra / 'manifest.json') for extra in extras},
            'consumed_regression': 'Original eight-case bass stress; never fitting/selection; no longer fresh.', 'preparation_sha256': digest(directory / 'plan.json'),
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
    training, validation = extend(training, validation, extras)
    preserve(output / 'expanded-baseline-audit.json', {'items': [
        {'id': item['id'], 'group': item['group'], 'corpus': item['corpus'],
         'baseline_bass': metrics(item['reference'], item['events'], item['seconds'])}
        for item in training + validation], 'no_test_metrics': True})
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
            context = [probability(checkpoint[0], item['x']) for item in validation]
            acoustic = [probability(checkpoint[1], item['x'][:, :26]) for item in validation]
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
    parser.add_argument('--run', default='bass-positive-consensus-v2')
    parser.add_argument('--extra', type=Path, action='append', default=[])
    args = parser.parse_args()
    train(args.directory, args.directory / args.run, args.extra)
