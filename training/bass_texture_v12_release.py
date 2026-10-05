"""Evaluate one frozen V12-retained wrong-pitch bass winner on consumed regressions.

These recordings are no longer fresh. Passing them is necessary but does not
authorize deployment or establish generalization on commercial songs.
"""

import argparse
import json
import pickle
from pathlib import Path

from bass_texture_v12_data import ACOUSTIC_NAMES, NAMES, VERSION, load_cached
from current_bass_v12_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from train_bass_texture_v12 import (
    GUARDIAN_POSITIVE_WEIGHT,
    GUARDIANS,
    PROFILES,
    STAGES,
    THRESHOLDS,
    contracts,
    predictions,
    score,
)


def load_frozen(run):
    plan = json.loads((run / 'plan.json').read_text())
    batch = json.loads((run / 'batch-selection.json').read_text())
    eligible = [row for row in batch['candidates'] if row['selected'] and row['validation_false_notes_removed'] > 0]
    if not eligible or not batch['selected']:
        raise ValueError('No positive frozen V12 validation winner')
    winner = max(eligible, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['trees']))
    if (winner != batch['winner'] or batch['test_used_for_selection'] or winner['test_used_for_selection']
            or batch['plan_sha256'] != digest(run / 'plan.json') or winner['plan_sha256'] != batch['plan_sha256']
            or plan['version'] != VERSION or plan['baseline_hashes'] != hashes() or plan['code_sha256'] != contracts()
            or plan['profiles'] != list(PROFILES) or plan['stages'] != list(STAGES)
            or plan['threshold_grid'] != list(THRESHOLDS) or plan['guardian_grid'] != list(GUARDIANS)
            or plan['guardian_positive_weight'] != GUARDIAN_POSITIVE_WEIGHT
            or plan['safety_factor'] != .5 or plan['feature_counts'] != [len(NAMES), len(ACOUSTIC_NAMES)]
            or plan['feature_names'] != list(NAMES) or plan['guardian_names'] != list(ACOUSTIC_NAMES)
            or winner['trees'] not in STAGES or winner['name'] not in {p['name'] for p in PROFILES}
            or winner['threshold'] not in {v * .5 for v in THRESHOLDS}
            or winner['guardian_threshold'] not in {v * .5 for v in GUARDIANS}):
        raise ValueError('Changed frozen V12 bass plan or winner')
    cache = Path(plan['baseline_cache_root'])
    if (digest(cache / 'manifest.json') != plan['baseline_cache_manifest_sha256']
            or digest(cache / 'plan.json') != plan['baseline_cache_plan_sha256']):
        raise ValueError('Changed frozen V12 baseline cache')
    source = run / winner['name']
    if json.loads((source / 'selection.json').read_text()) != winner:
        raise ValueError('Changed frozen V12 bass selection')
    if digest(source / 'validation.json') != winner['validation_sha256']:
        raise ValueError('Changed frozen V12 bass validation')
    validation = json.loads((source / 'validation.json').read_text())
    if (not validation['passes'] or not validation['per_recording'] or any(not row['passes'] for row in validation['per_recording'])
            or validation['threshold'] != winner['threshold'] or validation['guardian_threshold'] != winner['guardian_threshold']
            or validation['false_notes_removed'] != winner['validation_false_notes_removed']):
        raise ValueError('Failed frozen bass boundary validation')
    path = source / 'candidate.pickle'
    if digest(path) != winner['checkpoint_sha256']:
        raise ValueError('Changed frozen V12 bass checkpoint')
    with path.open('rb') as stream:
        models = pickle.load(stream)
    if ([model.n_features_in_ for model in models] != [len(NAMES), len(ACOUSTIC_NAMES)]
            or any(len(model._predictors) != winner['trees'] for model in models)):
        raise ValueError('Changed bass boundary model shape or depth')
    return plan, winner, models


def evaluate(run):
    destination = run / 'consumed-regression.json'
    if destination.exists():
        raise ValueError('Preserve consumed boundary evaluation; no runner-up or test tuning')
    plan, winner, models = load_frozen(run)
    items = load_cached(Path(plan['baseline_cache_root']), ('test',))
    context, acoustic = predictions(models, items)
    result = score(items, context, acoustic, winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  gate_sha256=digest(Path(__file__)), test_used_for_selection=False,
                  scope='All 156 predeclared consumed bass regressions, against actual V12 website notes. No fresh generalization claim; no deployment.')
    preserve(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    evaluate(parser.parse_args().run)
