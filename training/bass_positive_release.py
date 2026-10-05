"""One-shot bass release evaluation. No runner-up or threshold retuning on tests."""

import argparse
import json
import pickle
from pathlib import Path

from bass_training_data import load
from prepare_robust_training_stems import digest, preserve
from train_bass_positive_consensus import (
    GUARDIANS,
    POLICY,
    PROFILES,
    STAGES,
    THRESHOLDS,
    contracts,
    score,
)


def load_frozen(directory, run):
    plan = json.loads((run / 'plan.json').read_text())
    batch = json.loads((run / 'batch-selection.json').read_text())
    eligible = [row for row in batch['candidates'] if row['selected'] and row['validation_false_notes_removed'] > 0]
    if not eligible or not batch['selected']:
        raise ValueError('No positive frozen bass validation winner')
    winner = max(eligible, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['trees']))
    if (winner != batch['winner'] or batch['plan_sha256'] != digest(run / 'plan.json')
            or winner['plan_sha256'] != batch['plan_sha256'] or plan['policy'] != POLICY
            or winner['policy'] != POLICY or plan['profiles'] != list(PROFILES) or plan['stages'] != list(STAGES)
            or plan['threshold_grid'] != list(THRESHOLDS) or plan['guardian_grid'] != list(GUARDIANS)
            or plan['margin'] != .5 or winner['trees'] not in STAGES or plan['code_sha256'] != contracts()
            or plan['manifest_sha256'] != digest(directory / 'manifest.json')
            or plan['preparation_sha256'] != digest(directory / 'plan.json')
            or winner['threshold'] not in {v * .5 for v in THRESHOLDS}
            or winner['guardian_threshold'] not in {v * .5 for v in GUARDIANS}):
        raise ValueError('Changed frozen bass training plan or selection')
    for filename, expected in plan['additional_manifests'].items():
        if digest(Path(filename) / 'manifest.json') != expected:
            raise ValueError('Changed additional frozen bass data')
    source = run / winner['name']
    if json.loads((source / 'selection.json').read_text()) != winner:
        raise ValueError('Changed frozen bass candidate selection')
    validation = json.loads((source / 'validation.json').read_text())
    if (not validation['passes'] or not validation['per_recording']
            or any(not row['passes'] for row in validation['per_recording'])
            or validation['threshold'] != winner['threshold']
            or validation['guardian_threshold'] != winner['guardian_threshold']
            or validation['false_notes_removed'] != winner['validation_false_notes_removed']):
        raise ValueError('Changed or failed per-recording bass validation')
    path = source / 'candidate.pickle'
    if digest(path) != winner['checkpoint_sha256']:
        raise ValueError('Changed frozen bass checkpoint')
    with path.open('rb') as stream:
        models = pickle.load(stream)
    if ([model.n_features_in_ for model in models] != [52, 26]
            or any(len(model._predictors) != winner['trees'] for model in models)):
        raise ValueError('Changed bass model features or selected checkpoint depth')
    return winner, models


def require_report(run, winner, stage, positive=False):
    result = json.loads((run / (stage + '.json')).read_text())
    if (not result['passes'] or not result['per_recording']
            or result['checkpoint_sha256'] != winner['checkpoint_sha256']
            or result['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or result['threshold'] != winner['threshold']
            or result['guardian_threshold'] != winner['guardian_threshold']
            or any(not row['passes'] or not row['matched_references_preserved']
                   or not row['held_references_preserved'] or not row['reference_pitch_coverage_preserved']
                   for row in result['per_recording'])
            or (positive and result['false_notes_removed'] <= 0)):
        raise ValueError('Failed or changed frozen bass release evidence')
    return result


def evaluate(directory, run, data, stage):
    if stage not in ('held-regression', 'reserved-slakh'):
        raise ValueError('Unknown bass release stage')
    destination = run / (stage + '.json')
    if destination.exists():
        raise ValueError('Preserve consumed bass evaluation; do not retune on tests')
    winner, models = load_frozen(directory, run)
    if stage == 'reserved-slakh':
        require_report(run, winner, 'held-regression')
    items = load(data, 'test')
    if not items:
        raise ValueError('No untouched test evidence')
    manifest = json.loads((directory / 'manifest.json').read_text())
    fitted = {item['source_group'] for item in manifest['items']}
    plan = json.loads((run / 'plan.json').read_text())
    for filename in plan['additional_manifests']:
        other = json.loads((Path(filename) / 'manifest.json').read_text())
        fitted.update(item['source_group'] for item in other['items'])
    if any(item['source_group'] in fitted for item in items):
        raise ValueError('Test source groups overlap fitting or selection data')
    if stage == 'reserved-slakh':
        partition = json.loads((directory / 'plan.json').read_text())['project_partition']
        groups = {source for source, group in partition.items() if group == 'test'}
        if {item['source_group'] for item in items} != groups:
            raise ValueError('Reserved Slakh evaluation requires all four frozen groups')
    result = score(items, [models[0].predict_proba(item['x'])[:, 1] for item in items],
                   [models[1].predict_proba(item['x'][:, :26])[:, 1] for item in items],
                   winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  test_manifest_sha256=digest(data / 'manifest.json'), gate_sha256=digest(Path(__file__)),
                  scope='Original held/repeated/simultaneous bass stress' if stage == 'held-regression'
                  else 'Previously unscored Slakh groups; synthesized prototype, near-duplicate/pretraining overlap not ruled out')
    preserve(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--stage', required=True, choices=('held-regression', 'reserved-slakh'))
    args = parser.parse_args()
    evaluate(args.directory, args.run, args.data, args.stage)
