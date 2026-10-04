"""Additional frozen-candidate release gate: preserve reference-pitch coverage.

Attack and offset matches alone can miss loss of a later held-note fragment.
This audit cannot tune a candidate; it only accepts or rejects the frozen winner.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.note_context_model import residual_keep
from train_broad_consensus import digest, load_winner, validation_items
from train_left_consensus_v8 import prepare, probabilities, regression_items
from train_note_verifier import write


def reference_coverage(reference, events):
    values = []
    for start, end, pitch in reference:
        same = events[np.abs(events[:, 2] - pitch) <= .5]
        spans = sorted((max(start, first), min(end, last)) for first, last, *_ in same
                       if min(end, last) > max(start, first))
        total, stop = 0., start
        for first, last in spans:
            total += max(0., last - max(stop, first))
            stop = max(stop, last)
        values.append(total)
    return np.asarray(values)


def compare(reference, before, after):
    old, new = [reference_coverage(reference, events) for events in (before, after)]
    lost = np.maximum(0., old - new)
    return {'passes': not np.any(lost > 1e-6), 'lost_reference_seconds': float(lost.sum()),
            'references_with_lost_coverage': int(np.sum(lost > 1e-6))}


def audit(directory, source):
    destination = source / 'coverage-release-gate.json'
    if destination.exists():
        raise ValueError('Preserve completed coverage release evidence')
    winner, models = load_winner(source)
    regression = json.loads((source / 'regression.json').read_text())
    if (not regression['passes'] or regression['false_notes_removed'] <= 0
            or regression['checkpoint_sha256'] != winner['checkpoint_sha256']
            or regression['batch_selection_sha256'] != digest(source / 'batch-selection.json')):
        raise ValueError('Coverage release audit requires frozen positive regression improvement')
    torch.set_num_threads(2)
    rows = []
    for group, raw in [('validation', validation_items(directory, source)),
                       ('regression', regression_items(directory))]:
        for item in prepare(directory, raw, baseline_evidence=True, full_register=True):
            probability = probabilities(models, item['x'], winner['threshold'], winner['guardian_threshold'])
            keep = residual_keep(item['keep'], item['p'], item['shared'], probability,
                                 threshold=winner['threshold'], ceiling=1.)
            result = compare(item['reference'], item['events'][item['keep']], item['events'][keep])
            rows.append({'group': group, 'id': item['id'], **result})
    result = {'passes': all(row['passes'] for row in rows), 'per_recording': rows,
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'gate_code_sha256': digest(Path(__file__)),
              'scope': 'Frozen-candidate rejection gate; reference-pitch interval union, not audio perceptual accuracy; no tuning'}
    write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    audit(args.directory, args.source)
