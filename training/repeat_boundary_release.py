"""Frozen boundary-model loading and additional release evaluation, never tuning."""

import argparse
import json
import pickle
from pathlib import Path

import torch
from current_accompaniment_baseline import hashes, prepare
from train_broad_consensus import digest
from train_note_verifier import write
from train_repeat_boundaries import (
    GUARDIANS,
    PROFILES,
    THRESHOLDS,
    VERSION,
    annotate,
    predict,
    score,
)
from train_residual_verifier import cached_external


def load_frozen(run):
    batch = json.loads((run / 'batch-selection.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    eligible = [row for row in batch['candidates'] if row['selected'] and row['validation_false_notes_removed'] > 0]
    if not eligible or not batch['selected']:
        raise ValueError('No positive validation winner')
    winner = max(eligible, key=lambda row: row['validation_false_notes_removed'])
    if (batch['winner'] != winner or batch['plan_sha256'] != digest(run / 'plan.json')
            or winner['plan_sha256'] != batch['plan_sha256'] or plan['baseline_hashes'] != hashes()
            or winner['baseline_hashes'] != hashes() or plan['profiles'] != list(PROFILES)
            or plan['version'] != VERSION or plan['safety_factor'] != .5
            or plan['threshold_grid'] != list(THRESHOLDS) or plan['guardian_grid'] != list(GUARDIANS)
            or winner['threshold'] not in {v * .5 for v in THRESHOLDS}
            or winner['guardian_threshold'] not in {v * .5 for v in GUARDIANS}):
        raise ValueError('Changed frozen boundary selection')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Changed frozen boundary code')
    for filename, expected in plan['extra_manifest_sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Changed fitting manifest')
    source = run / winner['name']
    if json.loads((source / 'selection.json').read_text()) != winner:
        raise ValueError('Changed frozen candidate selection')
    checkpoint = source / 'candidate.pickle'
    if digest(checkpoint) != winner['checkpoint_sha256']:
        raise ValueError('Changed frozen candidate weights')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    if [m.n_features_in_ for m in models] != [152, 112]:
        raise ValueError('Unexpected boundary feature contract')
    return winner, models


def require_report(run, winner, stage, positive=False):
    report = json.loads((run / (stage + '.json')).read_text())
    if (not report['passes'] or not report['per_recording']
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or report['threshold'] != winner['threshold']
            or report['guardian_threshold'] != winner['guardian_threshold']
            or any(not row['passes'] or not row['matched_references_preserved']
                   or not row['held_references_preserved'] or not row['reference_pitch_coverage_preserved']
                   or not row['pitch_time_union_preserved'] for row in report['per_recording'])
            or (positive and report['false_notes_removed'] <= 0)):
        raise ValueError('Failed or changed frozen boundary release evidence')
    return report


def evaluate(directory, run, manifest, stage):
    if stage not in ('current-piano-regression', 'fresh-stress'):
        raise ValueError('Unknown release stage')
    output = run / (stage + '.json')
    if output.exists():
        raise ValueError('Preserve consumed evaluation; never tune the winner on it')
    winner, models = load_frozen(run)
    require_report(run, winner, 'regression', positive=True)
    if stage == 'fresh-stress':
        require_report(run, winner, 'current-piano-regression')
    raw = list(cached_external(directory, manifest, directory))
    if not raw or any(item.get('group') != 'test' for item in raw):
        raise ValueError('Release evaluation must use reserved test sources')
    torch.set_num_threads(2)
    items = annotate(prepare(directory, raw))
    result = score(items, predict(models, items), winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  manifest_sha256=digest(manifest), gate_code_sha256=digest(Path(__file__)),
                  scope='Unscored MP3/separation conditions, previously seen test performers/compositions'
                  if stage == 'fresh-stress' else 'Latest consumed V8 piano passages; not fresh evaluation')
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--stage', required=True, choices=('current-piano-regression', 'fresh-stress'))
    args = parser.parse_args()
    evaluate(args.directory, args.run, args.manifest, args.stage)
