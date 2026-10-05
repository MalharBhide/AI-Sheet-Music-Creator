"""Fit harmonic attack/release heads on actual shipped V13 survivors."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from bass_harmonic_v13_data import ACOUSTIC_NAMES, NAMES, VERSION, acoustic_view, collect
from bass_predictions import probability
from current_bass_v13_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_bass_boundaries import contracts as baseline_contracts
from train_bass_consensus import GUARDIANS, STAGES, THRESHOLDS, at_stage, matrix, select
from train_bass_consensus import score as score

PROFILES = (
    {'name': 'harmonic', 'max_iter': 600, 'positive_weight': 24., 'min_samples_leaf': 25,
     'max_leaf_nodes': 31, 'l2_regularization': 8.},
    {'name': 'hold_guarded', 'max_iter': 600, 'positive_weight': 40., 'min_samples_leaf': 50,
     'max_leaf_nodes': 15, 'l2_regularization': 12.},
)
GUARDIAN_POSITIVE_WEIGHT = 40.


def contracts():
    root = Path(__file__).resolve().parents[1]
    return {**baseline_contracts(), **{name: digest(root / name) for name in (
        'training/train_bass_harmonic_v13.py', 'training/bass_harmonic_v13_data.py',
        'training/harmonic_candidate_features.py', 'training/fresh_bass_harmonic_v13.py',
        'training/prepare_bass_v13_baseline.py', 'training/current_bass_v13_baseline.py',
        'training/bass_harmonic_v13_release.py', 'training/bass_residual_coverage_data.py',
        'training/pitch_support_targets.py', 'training/cache_note_verifier.py',
        'backend/app/services/bass_texture.py', 'backend/app/services/candidate_relations.py')}}


def predictions(models, items):
    return ([probability(models[0], item['x']) for item in items],
            [probability(models[1], acoustic_view(item['x'])) for item in items])


def train(directory, output):
    output.mkdir(exist_ok=False)
    cache_manifest = json.loads((directory / 'manifest.json').read_text())
    fitted = {i['source_group'] for i in cache_manifest['items'] if i['group'] in ('train', 'validation')}
    regression = [i for i in cache_manifest['items'] if i['group'] == 'test']
    if not regression or any(i['source_group'] in fitted for i in regression):
        raise ValueError('V13 retained consumed regression overlaps fitting')
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'code_sha256': contracts(),
            'baseline_cache_root': str(directory),
            'baseline_cache_manifest_sha256': digest(directory / 'manifest.json'),
            'baseline_cache_plan_sha256': digest(directory / 'plan.json'),
            'preparation_evidence_sha256': digest(directory / 'preparation-evidence.json'),
            'profiles': list(PROFILES), 'guardian_positive_weight': GUARDIAN_POSITIVE_WEIGHT,
            'stages': list(STAGES), 'threshold_grid': list(THRESHOLDS),
            'guardian_grid': list(GUARDIANS), 'safety_factor': .5, 'feature_counts': [len(NAMES), len(ACOUSTIC_NAMES)],
            'feature_names': list(NAMES), 'guardian_names': list(ACOUSTIC_NAMES), 'sklearn_version': sklearn.__version__,
            'confidence_features': 'No fitted-model probabilities or truth; 84 observed acoustic/harmonic candidate relationships, guardian all 52 per-note attack/release channels.',
            'baseline_metric_key': 'baseline_bass means the actual V13 website after all four bass stages, never raw BP, V10 or V11.',
            'label_policy': 'Positive KEEP supervision for every annotated pitch overlap exceeding the frozen 1e-6-second gate tolerance, plus matched key attacks/releases. Features and evaluation truth unchanged.',
            'selection': 'Validation only: maximum safe false-note reduction, lower loss, fewer trees, fixed profile order.',
            'gates': 'Every matched attack and offset, each supplied pitch interval and per-recording false-note nonincrease, raw and half-margin.',
            'scope': 'Wrong pitches retained by balanced V13 bass; previous deletions cannot return; note times/pitches of retained events unchanged. No user uploads or scores.'}
    preserve(output / 'plan.json', plan)
    training, validation = collect(directory)
    audit = [{'id': item['id'], 'group': item['group'], 'corpus': item['corpus'],
              'retained_v12': item.get('retained_v12'), 'retained_v13': len(item['events']),
              'eligible': int(item['eligible'].sum()), 'positive': int(item['y'][item['eligible'] & item['mask']].sum()),
              'negative': int(((item['y'] == 0) & item['eligible'] & item['mask']).sum()),
              'held_support_promoted': int(item['promoted'].sum()),
              'protected_offset_assignments': item['protected_offset_assignments'],
              'partial_pitch_support_promoted': item['partial_pitch_support_promoted']} for item in training + validation]
    preserve(output / 'supervision.json', {'items': audit, 'no_test_metrics': True})
    vx, vy, vw, _ = matrix(validation, 1.)
    selections = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        settings = {key: value for key, value in profile.items() if key not in ('name', 'positive_weight')}
        settings.update(learning_rate=.04, early_stopping=False, random_state=262701 + index * 2)
        guardian_settings = {**settings, 'max_leaf_nodes': 7, 'random_state': 262702 + index * 2}
        gx, gy, gw, _ = matrix(training, GUARDIAN_POSITIVE_WEIGHT)
        np.testing.assert_array_equal(gy, y)
        np.testing.assert_array_equal(gx, x)
        preserve(destination / 'run.json', {'settings': settings, 'guardian_settings': guardian_settings,
                 'training_counts': counts, 'training_clips': len(training), 'validation_clips': len(validation),
                 'training_events': len(y), 'positive_weight': profile['positive_weight'], 'guardian_positive_weight': GUARDIAN_POSITIVE_WEIGHT,
                 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, curves = [], {}
        for name, config, tx, test_x, sample_weight in [('relation', settings, x, vx, weight),
                                                      ('guardian', guardian_settings, acoustic_view(gx), acoustic_view(vx), gw)]:
            print(f'Fitting V13 harmonic-relationship bass {profile["name"]} {name}: {len(y)} events', flush=True)
            model = HistGradientBoostingClassifier(**config).fit(tx, y, sample_weight=sample_weight)
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, p[:, 1], sample_weight=vw, labels=[0., 1.]))}
                            for step, p in enumerate(model.staged_predict_proba(test_x), 1) if step in STAGES]
            models.append(model)
        preserve(destination / 'learning-curves.json', curves)
        best, stages, memo = None, [], {}
        for trees in [step for step in STAGES if step <= profile['max_iter']]:
            checkpoint = at_stage(models, trees)
            context, acoustic = predictions(checkpoint, validation)
            chosen, search = select(validation, context, acoustic, memo)
            loss = sum(next(row['validation_log_loss'] for row in curves[name] if row['trees'] == trees)
                       for name in ('relation', 'guardian')) / 2
            stage = {'trees': trees, 'validation_log_loss': loss, 'selected': chosen is not None,
                     'false_notes_removed': chosen['false_notes_removed'] if chosen else 0}
            stages.append(stage)
            preserve(destination / f'search-{trees}.json', search)
            if chosen:
                key = (chosen['false_notes_removed'], -loss, -trees)
                if best is None or key > best[0]:
                    best = key, checkpoint, chosen, trees, loss
            print(json.dumps({'profile': profile['name'], **stage}), flush=True)
        preserve(destination / 'stages.json', stages)
        with (destination / 'candidate.pickle').open('wb') as stream:
            pickle.dump(best[1] if best else models, stream)
        if best:
            preserve(destination / 'validation.json', best[2])
        selection = {'name': profile['name'], 'selected': best is not None,
                     'validation_false_notes_removed': best[2]['false_notes_removed'] if best else 0,
                     'threshold': best[2]['threshold'] if best else 0.,
                     'guardian_threshold': best[2]['guardian_threshold'] if best else 0.,
                     'trees': best[3] if best else profile['max_iter'], 'validation_log_loss': best[4] if best else None,
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
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    train(args.directory, args.output)
