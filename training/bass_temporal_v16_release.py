"""One validation-only temporal winner; consumed evaluation cannot tune it."""

import argparse
import json
from pathlib import Path

from app.services.bass_harmonic import BassHarmonic
from current_bass_v15_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from train_bass_consensus import GUARDIANS, THRESHOLDS, score
from train_bass_temporal_v16 import (
    FRESH_SEEDS,
    PROFILES,
    STAGES,
    checkpoint_model,
    contracts,
    predictions,
)


def load_frozen(run):
    plan = json.loads((run / 'plan.json').read_text())
    batch = json.loads((run / 'batch-selection.json').read_text())
    eligible = [row for row in batch['candidates'] if row['selected'] and row['validation_false_notes_removed'] > 0]
    if not eligible or not batch['selected']:
        raise ValueError('No safe positive temporal validation winner')
    winner = max(eligible, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['epoch']))
    if (winner != batch['winner'] or batch['test_used_for_selection'] or winner['test_used_for_selection']
            or plan['baseline_hashes'] != hashes() or plan['code_sha256'] != contracts()
            or plan['profiles'] != list(PROFILES) or plan['stages'] != list(STAGES)
            or plan['threshold_grid'] != list(THRESHOLDS) or plan['guardian_grid'] != list(GUARDIANS)
            or plan['fresh_seeds'] != list(FRESH_SEEDS) or plan['safety_factor'] != .5
            or batch['plan_sha256'] != digest(run / 'plan.json') or winner['plan_sha256'] != batch['plan_sha256']
            or winner['threshold'] not in {v * .5 for v in THRESHOLDS}
            or winner['guardian_threshold'] not in {v * .5 for v in GUARDIANS} or winner['epoch'] not in STAGES):
        raise ValueError('Changed frozen temporal plan or winner')
    for key in ('baseline', 'piano'):
        folder = Path(plan[key + '_root'])
        if (digest(folder / 'manifest.json') != plan[key + '_manifest_sha256']
                or digest(folder / 'plan.json') != plan[key + '_plan_sha256']):
            raise ValueError('Changed expanded temporal fitting data')
    if digest(Path(plan['piano_root']) / 'source-audit.json') != plan['piano_audit_sha256']:
        raise ValueError('Changed real piano exclusions')
    source = run / winner['name']
    if (json.loads((source / 'selection.json').read_text()) != winner
            or digest(source / 'validation.json') != winner['validation_sha256']
            or digest(run / winner['checkpoint']) != winner['checkpoint_sha256']):
        raise ValueError('Changed temporal checkpoint or validation')
    validation = json.loads((source / 'validation.json').read_text())
    if (not validation['passes'] or validation['false_notes_removed'] != winner['validation_false_notes_removed']
            or any(not row['passes'] for row in validation['per_recording'])
            or (validation['threshold'], validation['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])):
        raise ValueError('Failed frozen temporal preservation gates')
    model, normalizer = checkpoint_model(run / winner['checkpoint'])
    return plan, winner, model, normalizer


def evaluate(run):
    from bass_temporal_v15_data import baseline_items

    plan, winner, model, normalizer = load_frozen(run)
    items = baseline_items(Path(plan['baseline_root']), ('test',))
    if len(items) != 252 or any(i['group'] != 'test' for i in items):
        raise ValueError('Incomplete or illegal consumed expanded temporal evaluation')
    guardian = BassHarmonic().models[1]
    result = score(items, predictions(model, normalizer, items), [guardian.probability(i['base_x']) for i in items],
                   winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'], batch_selection_sha256=digest(run / 'batch-selection.json'),
                  test_used_for_selection=False, regression_manifest_sha256=plan['baseline_manifest_sha256'],
                  scope='All 252 consumed regressions against actual V15 notes; no runner-up or threshold tuning, no deployment. Expanded real-piano regression and first-pass fixtures still required.')
    preserve(run / 'consumed-regression.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    evaluate(args.run)
