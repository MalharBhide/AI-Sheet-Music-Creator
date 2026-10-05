"""Evaluate the single frozen attack candidate on consumed singer regressions."""

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
from app.services.melody_decoder import decode
from app.services.vocal_melody import CHECKPOINT
from prepare_robust_training_stems import digest, preserve
from train_attack_timing_v2 import evaluate
from train_context_melody import baseline_model, load_vocalset
from train_melody import load_data, predict
from train_melody_timing import boundary_features


def frozen(run):
    plan = json.loads((run / 'plan.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    reports = json.loads((run / 'validation.json').read_text())
    candidates = [r for r in reports if r['passes'] and r['changed_attacks'] > 0
                  and all(v['after_mae'] <= v['before_mae'] + 1e-12 for v in r['aggregate'].values())
                  and sum(v['before_mae']-v['after_mae'] for v in r['aggregate'].values()) > 1e-9]
    winner = min(candidates, key=lambda r: sum(v['after_mae'] for v in r['aggregate'].values())) if candidates else None
    if (not selection['selected'] or winner is None
            or selection['winner'] != {k:v for k,v in winner.items() if k != 'per_recording'}
            or selection['plan_sha256'] != digest(run / 'plan.json') or selection['test_used']
            or selection['checkpoint_sha256'] != digest(run / 'candidate.pickle')
            or plan['baseline_sha256'] != digest(CHECKPOINT)):
        raise ValueError('Changed or failed frozen timing candidate')
    root = Path(__file__).resolve().parents[1]
    if any(digest(root / name) != sha for name, sha in plan['source_sha256'].items()):
        raise ValueError('Changed frozen timing producer')
    return plan, selection


def regression(directory, run):
    if (run / 'consumed-regression.json').exists():
        raise ValueError('No retuning, overwrite or replacement consumed check')
    plan, selection = frozen(run)
    if any(digest(directory / name) != sha for name, sha in plan['split_sha256'].items()):
        raise ValueError('Changed singer partitions')
    # This pickle is created locally by this experiment and checked by SHA before
    # deserialization. It is never accepted from uploads or loaded by the website.
    with (run / 'candidate.pickle').open('rb') as stream:
        candidate = pickle.load(stream)
    torch.set_num_threads(2)
    baseline, checkpoint = baseline_model()
    split = json.loads((directory / 'melody-v1/split.json').read_text())
    collections = [('vocadito', load_data(directory, split['tracks']['test'])),
                   ('vocalset', load_vocalset(directory, 'test'))]
    items = []
    for corpus, rows in collections:
        for item, (logits, attacks) in zip(rows, predict(baseline, rows), strict=True):
            events = decode(logits, attacks, **checkpoint['decoder'])
            reference = np.array([[s, s+d, float(librosa.hz_to_midi(hz))] for s, hz, d in item['A1']]).reshape(-1, 3)
            items.append({'id': corpus + '-' + str(item['id']), 'corpus': corpus, 'events': events,
                          'reference': reference, 'onset_x': boundary_features(item['x'], events, 0),
                          'duration': len(item['x']) / 50})
    if len(items) != 54:
        raise ValueError('Incomplete consumed singer regression')
    winner = selection['winner']
    report = evaluate(items, candidate, winner['threshold'], winner['scale'])
    report.update(checkpoint_sha256=selection['checkpoint_sha256'],
                  selection_sha256=digest(run / 'selection.json'),
                  baseline_sha256=plan['baseline_sha256'], no_retuning=True, now_consumed=True,
                  scope='All 54 previously consumed singer tests. Frozen single winner; no user audio. Features reused; baseline model inference only, no score generation or independent-human accuracy claim.')
    frozen(run)
    preserve(run / 'consumed-regression.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    regression(args.directory, args.run)
