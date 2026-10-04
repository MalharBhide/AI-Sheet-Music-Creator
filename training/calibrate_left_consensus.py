"""Select separate view thresholds on validation, with unchanged release gates."""

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import torch
from train_left_relations_v7 import combine_scores, hashes, prepare, score
from train_note_verifier import load_data, write
from train_residual_verifier import THRESHOLDS, cached_external

GUARDIAN_THRESHOLDS = (.1, .2, .4, .6)


def calibrate(directory, source, output, extras):
    output.mkdir(exist_ok=False)
    original = json.loads((source / 'selection.json').read_text())
    source_run = json.loads((source / 'run.json').read_text())
    checkpoint = source / 'candidate.pickle'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (original['baseline_hashes'] != hashes() or digest != original['checkpoint_sha256']
            or not original['consensus'] or original['policy'] != 'refinement'):
        raise ValueError('Wrong baseline or consensus checkpoint')
    if sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in extras) != sorted(
            source_run['extra_manifest_sha256'].values()):
        raise ValueError('Calibration must use the same frozen validation manifests')
    run = {**source_run, 'source_checkpoint_sha256': digest,
           'calibration': 'separate-view-thresholds-v1', 'guardian_threshold_grid': GUARDIAN_THRESHOLDS,
           'threshold_grid': THRESHOLDS, 'selection_safety_factor': .5,
           'test_used_for_selection': False}
    write(output / 'run.json', run)  # Freeze the grid before inspecting scores.
    shutil.copy2(checkpoint, output / 'candidate.pickle')
    with checkpoint.open('rb') as stream:
        model = pickle.load(stream)  # Hash-verified local fit; never deployed as pickle.
    items = load_data(directory, 'validation')
    for path in extras:
        items.extend(item for item in cached_external(directory, path, directory) if item['group'] == 'validation')
    torch.set_num_threads(2)
    items = prepare(directory, items)
    context = [model[0][1].predict_proba(item['x'])[:, 1] for item in items]
    acoustic = [model[1][1].predict_proba(item['x'][:, :52])[:, 1] for item in items]
    search, best, memo = [], None, {}
    for guardian in GUARDIAN_THRESHOLDS:
        for threshold in THRESHOLDS:
            probabilities = [combine_scores(c, a, threshold, guardian) for c, a in zip(context, acoustic, strict=True)]
            raw, margin = [score(items, probabilities, at, memo) for at in (threshold, threshold * .5)]
            # Scaling is unchanged when both thresholds are halved together.
            search.append({'guardian_raw': guardian, 'raw': {k: v for k, v in raw.items() if k != 'per_recording'},
                           'margin': {k: v for k, v in margin.items() if k != 'per_recording'}})
            if raw['passes'] and margin['passes'] and margin['false_notes_removed'] > 0 and (
                    best is None or margin['false_notes_removed'] > best[0]['false_notes_removed']):
                best = margin, guardian * .5
    result, guardian = best if best else (score(items, context, 0.), 0.)
    selection = {**original, 'selected': best is not None, 'threshold': result['threshold'],
                 'guardian_threshold': guardian, 'calibration': run['calibration'],
                 'selection': 'Separate validation-only view thresholds; every recording passes at raw and half-margin values',
                 'test_not_evaluated': True}
    write(output / 'search.json', search)
    write(output / 'validation.json', result)
    write(output / 'selection.json', selection)
    print(json.dumps({**selection, 'validation_false_notes_removed': result['false_notes_removed']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--extra', type=Path, action='append', default=[])
    args = parser.parse_args()
    calibrate(args.directory, args.source, args.output, args.extra)
