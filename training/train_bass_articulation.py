"""Fit release-aware bass boundary heads, keeping historical experiments frozen."""

import argparse
import json
import pickle
from pathlib import Path

import sklearn
from bass_articulation_labels import VERSION, annotate
from bass_boundary_evidence import acoustic_view
from current_bass_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from train_bass_boundaries import (
    GUARDIANS,
    PROFILES,
    STAGES,
    THRESHOLDS,
    at_stage,
    collect,
    matrix,
    predictions,
    select,
)
from train_bass_boundaries import contracts as previous_contracts
from train_bass_boundaries import score as score


def contracts():
    return {**previous_contracts(), **{'training/' + name: digest(Path(__file__).with_name(name)) for name in (
        'train_bass_articulation.py', 'bass_articulation_labels.py', 'bass_articulation_release.py')}}


def train(roots, output, regressions):
    output.mkdir(exist_ok=False)
    fitted_sources = {item['source_group'] for root in roots
                      for item in json.loads((root / 'manifest.json').read_text())['items']}
    ids = set()
    for root in regressions:
        manifest = json.loads((root / 'manifest.json').read_text())
        if not manifest['items'] or any(item['group'] != 'test' or item['source_group'] in fitted_sources
                                        for item in manifest['items']):
            raise ValueError('Consumed regression cannot enter articulation fitting')
        for item in manifest['items']:
            if item['id'] in ids:
                raise ValueError('Duplicate consumed articulation regression')
            ids.add(item['id'])
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'code_sha256': contracts(),
            'manifests': {str(root): digest(root / 'manifest.json') for root in roots},
            'consumed_regression_manifests': {str(root): digest(root / 'manifest.json') for root in regressions},
            'profiles': list(PROFILES), 'stages': list(STAGES), 'threshold_grid': list(THRESHOLDS),
            'guardian_grid': list(GUARDIANS), 'safety_factor': .5, 'feature_counts': [112, 60],
            'sklearn_version': sklearn.__version__,
            'change': 'Fitting annotations additionally veto removing a matched key attack or offset; unchanged label-free features, model profiles, seeds and thresholds.',
            'selection': 'Validation-only maximum safe false attack reduction, then lower log loss, fewer trees and fixed profile order.',
            'gates': 'Per-recording matched attacks/offsets, pitch support, exact full detected pitch-time union and false-note nonincrease.',
            'scope': 'Dataset caches only; no user uploads, score regeneration or job creation. No deployment without independent release checks.'}
    preserve(output / 'plan.json', plan)
    training, validation = [annotate(items) for items in collect(roots)]
    rows = [{'id': item['id'], 'group': item['group'], 'corpus': item['corpus'],
             'pairs': len(item['pairs']), 'protected_key_offsets': int(item['protected_key_offsets'].sum()),
             'protected_key_attacks': int(item['protected_key_attacks'].sum()),
             'positive_boundaries': int(item['boundary_y'][item['boundary_mask']].sum()),
             'split_holds': int(((item['boundary_y'] == 0) & item['boundary_mask']).sum()),
             'ambiguous': int((~item['boundary_mask']).sum())} for item in training + validation]
    preserve(output / 'supervision.json', {'items': rows, 'counts': {group: {
        key: sum(row[key] for row in rows if row['group'] == group)
        for key in ('pairs', 'protected_key_offsets', 'protected_key_attacks', 'positive_boundaries', 'split_holds', 'ambiguous')}
        for group in ('train', 'validation')}})
    vx, vy, vw, _ = matrix(validation, 1.)
    selections = []
    for index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        x, y, weight, counts = matrix(training, profile['positive_weight'])
        config = {k: v for k, v in profile.items() if k not in ('name', 'positive_weight')}
        config.update(learning_rate=.04, early_stopping=False, random_state=261141 + index * 2)
        guardian_config = {**config, 'max_leaf_nodes': 7, 'random_state': 261142 + index * 2}
        counts = [{**{key: value for key, value in row.items() if key != 'repeats'},
                   'protected_boundaries': row['repeats']} for row in counts]
        preserve(destination / 'run.json', {'settings': config, 'guardian_settings': guardian_config,
                 'training_counts': counts, 'training_events': len(y), 'training_clips': len(training),
                 'validation_clips': len(validation), 'plan_sha256': digest(output / 'plan.json'),
                 'test_used_for_selection': False})
        models, curves = [], {}
        for name, settings, tx, test_x in [('repeat', config, x, vx),
                                         ('guardian', guardian_config, acoustic_view(x), acoustic_view(vx))]:
            print(f'Fitting release-aware bass {profile["name"]} {name}: {len(y)} pairs', flush=True)
            model = HistGradientBoostingClassifier(**settings).fit(tx, y, sample_weight=weight)
            curves[name] = [{'trees': step, 'validation_log_loss': float(log_loss(vy, prediction[:, 1],
                            sample_weight=vw, labels=[0., 1.]))}
                            for step, prediction in enumerate(model.staged_predict_proba(test_x), 1) if step in STAGES]
            models.append(model)
        preserve(destination / 'learning-curves.json', curves)
        best, stage_rows, memo = None, [], {}
        for trees in [step for step in STAGES if step <= profile['max_iter']]:
            staged = at_stage(models, trees)
            chosen, search = select(validation, predictions(staged, validation), memo)
            loss = sum(next(row['validation_log_loss'] for row in curves[name] if row['trees'] == trees)
                       for name in ('repeat', 'guardian')) / 2
            row = {'trees': trees, 'validation_log_loss': loss, 'selected': chosen is not None,
                   'false_notes_removed': chosen['false_notes_removed'] if chosen else 0}
            stage_rows.append(row)
            preserve(destination / f'search-{trees}.json', search)
            if chosen:
                key = (chosen['false_notes_removed'], -loss, -trees)
                if best is None or key > best[0]:
                    best = key, staged, chosen, trees, loss
            print(json.dumps({'profile': profile['name'], **row}), flush=True)
        preserve(destination / 'stages.json', stage_rows)
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
