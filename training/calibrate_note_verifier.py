"""Conservatively recalibrate frozen weights on validation data only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from train_note_verifier import evaluate, frozen, load_data, passes_guard, probabilities, write


def calibrate(directory, source, output_name, recall_margin):
    output = directory / output_name
    if output.exists():
        raise ValueError('Preserve earlier calibration attempts')
    model, checkpoint, source_digest = frozen(directory, source)
    items = load_data(directory, 'validation')
    scores = probabilities(model, items, checkpoint['mean'].numpy(), checkpoint['scale'].numpy())
    trials, best, selected = [], -1., None
    for threshold in np.arange(0., .801, .02):
        result = evaluate(items, scores, float(threshold))
        eligible = passes_guard(result) and all(row['trained']['recall'] >= row['baseline']['recall'] - recall_margin
                                              for row in result['aggregate'].values())
        f1 = float(np.mean([row['trained']['f1'] for row in result['aggregate'].values()]))
        trials.append({'threshold': float(threshold), 'eligible': eligible, 'aggregate': result['aggregate']})
        if eligible and f1 > best:
            best, selected, chosen = f1, result, float(threshold)
    output.mkdir()
    checkpoint['threshold'] = chosen
    torch.save(checkpoint, output / 'candidate.pt')
    digest = hashlib.sha256((output / 'candidate.pt').read_bytes()).hexdigest()
    write(output / 'selection.json', {'checkpoint_sha256': digest, 'threshold': chosen,
        'source_checkpoint_sha256': source_digest, 'validation_recall_margin': recall_margin,
        'selection': 'Maximum macro-corpus validation F1 with per-corpus recall loss <= configured margin and nondecreasing F1. Weights unchanged.',
        'limitations': 'Recalibration follows observed domain failures; prior tests are consumed regression checks, not fresh tests.',
        'fresh_segment_test': 'Oxford audio 30–60s from all eight recordings; clocks fixed from 0–15s; same recording/performer, different events. No new performer-independence claim.'})
    write(output / 'threshold-search.json', trials)
    write(output / 'validation.json', selected)
    (output / 'run.json').write_bytes((directory / source / 'run.json').read_bytes())
    (output / 'split.json').write_bytes((directory / 'note-verifier-split.json').read_bytes())
    print(json.dumps({'threshold': chosen, 'sha256': digest}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source', default='note-verifier-v2')
    parser.add_argument('--output', default='note-verifier-v2-conservative')
    parser.add_argument('--recall-margin', type=float, default=.0025)
    args = parser.parse_args()
    calibrate(args.directory, args.source, args.output, args.recall_margin)
