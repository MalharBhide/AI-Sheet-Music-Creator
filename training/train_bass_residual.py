"""Learn wrong-pitch/harmonic residuals against exact shipped V10 bass decisions."""

import argparse
import json
import pickle
from pathlib import Path

import sklearn
from bass_predictions import probability
from bass_residual_evidence import ACOUSTIC_NAMES, NAMES, VERSION, acoustic_view, collect
from current_bass_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_bass_boundaries import contracts as baseline_contracts
from train_bass_consensus import GUARDIANS, PROFILES, STAGES, THRESHOLDS, at_stage, matrix, select
from train_bass_consensus import score as score


def contracts():
    root = Path(__file__).resolve().parents[1]
    return {**baseline_contracts(), **{name: digest(root / name) for name in (
        'training/train_bass_residual.py', 'training/bass_residual_evidence.py',
        'training/bass_residual_release.py', 'training/pitch_support_targets.py',
        'backend/app/services/candidate_relations.py')}}


def predictions(models, items):
    return ([probability(models[0], item['x']) for item in items],
            [probability(models[1], acoustic_view(item['x'])) for item in items])


def train(roots, output, regressions):
    output.mkdir(exist_ok=False)
    fitted = {item['source_group'] for root in roots for item in json.loads((root / 'manifest.json').read_text())['items']}
    regression_ids = set()
    for root in regressions:
        items = json.loads((root / 'manifest.json').read_text())['items']
        if not items or any(item['group'] != 'test' or item['source_group'] in fitted for item in items):
            raise ValueError('Residual bass regression overlaps fitting')
        for item in items:
            if item['id'] in regression_ids:
                raise ValueError('Duplicate residual bass regression')
            regression_ids.add(item['id'])
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'code_sha256': contracts(),
            'manifests': {str(root): digest(root / 'manifest.json') for root in roots},
            'consumed_regression_manifests': {str(root): digest(root / 'manifest.json') for root in regressions},
            'profiles': list(PROFILES), 'stages': list(STAGES), 'threshold_grid': list(THRESHOLDS),
            'guardian_grid': list(GUARDIANS), 'safety_factor': .5, 'feature_counts': [len(NAMES), len(ACOUSTIC_NAMES)],
            'feature_names': list(NAMES), 'guardian_names': list(ACOUSTIC_NAMES), 'sklearn_version': sklearn.__version__,
            'baseline_metric_key': 'baseline_bass means the shipped V10 retained candidates, never raw Basic Pitch.',
            'selection': 'Validation only: maximum safe false-note reduction, lower loss, fewer trees, fixed profile order.',
            'gates': 'Every matched attack and offset, each supplied pitch interval and per-recording false-note nonincrease, raw and half-margin.',
            'scope': 'Wrong pitches retained by balanced V10 bass; previous deletions cannot return; note times/pitches of retained events unchanged. No user uploads or scores.'}
    preserve(output / 'plan.json', plan)
    training, validation = collect(roots)
    audit = [{'id': item['id'], 'group': item['group'], 'corpus': item['corpus'],
              'raw_candidates': item['raw_count'], 'retained_v10': len(item['events']),
              'eligible': int(item['eligible'].sum()), 'positive': int(item['y'][item['eligible'] & item['mask']].sum()),
              'negative': int(((item['y'] == 0) & item['eligible'] & item['mask']).sum()),
              'held_support_promoted': int(item['promoted'].sum())} for item in training + validation]
    preserve(output / 'supervision.json', {'items': audit, 'no_test_metrics': True})
    vx, vy, vw, _ = matrix(validation, 1.)
    selections = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        settings = {key: value for key, value in profile.items() if key not in ('name', 'positive_weight')}
        settings.update(learning_rate=.04, early_stopping=False, random_state=261251 + index * 2)
        guardian_settings = {**settings, 'max_leaf_nodes': 7, 'random_state': 261252 + index * 2}
        preserve(destination / 'run.json', {'settings': settings, 'guardian_settings': guardian_settings,
                 'training_counts': counts, 'training_clips': len(training), 'validation_clips': len(validation),
                 'training_events': len(y), 'positive_weight': profile['positive_weight'],
                 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        models, curves = [], {}
        for name, config, tx, test_x in [('relation', settings, x, vx),
                                       ('guardian', guardian_settings, acoustic_view(x), acoustic_view(vx))]:
            print(f'Fitting V10 bass residual {profile["name"]} {name}: {len(y)} events', flush=True)
            model = HistGradientBoostingClassifier(**config).fit(tx, y, sample_weight=weight)
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
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--regression', type=Path, action='append', default=[])
    args = parser.parse_args()
    train(args.roots, args.output, args.regression)
