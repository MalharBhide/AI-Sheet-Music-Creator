"""Training-only labels for genuine repeated attacks versus split held notes.

Only exact shared interior V7-retained events enter a boundary example. Labels
come from existing note annotations, never model predictions or song identity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train_broad_consensus import collect, digest
from train_left_consensus_v8 import hashes, prepare
from train_note_verifier import write

VERSION = 'repeat-boundary-supervision-v1'


def boundaries(item):
    """Return adjacent same-key pairs only when observed timing is near-contiguous."""
    events = item['events']
    previous, pairs = {}, []
    for index in np.argsort(events[:, 0], kind='stable'):
        if not item['keep'][index]:
            continue
        start, _, pitch, _ = events[index]
        earlier = previous.get(int(pitch))
        previous[int(pitch)] = int(index)
        if earlier is None or not (item['shared'][earlier] and item['shared'][index]):
            continue
        gap = start - events[earlier, 1]
        if not (-1e-6 <= gap <= .03 and .15 <= start - events[earlier, 0] <= 2.):
            continue
        pairs.append((earlier, int(index)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def labels(reference, events, pairs):
    y, mask = np.zeros(len(pairs), dtype=np.float32), np.zeros(len(pairs), dtype=bool)
    for index, (before, after) in enumerate(pairs):
        first, current = events[before], events[after]
        same = reference[np.abs(reference[:, 2] - current[2]) <= .5]
        if np.any(np.abs(same[:, 0] - current[0]) <= .05):
            y[index], mask[index] = 1., True
            continue
        if np.any(np.abs(same[:, 0] - current[0]) <= .15):
            continue
        # A labeled continuous hold must cover the entire observed pair. Late
        # isolated detections, real rests and extended tails remain ambiguous.
        hold = (same[:, 0] <= first[0] + .05) & (same[:, 1] >= max(first[1], current[1]) - .05)
        hold &= (same[:, 0] + .15 < current[0]) & (current[0] < same[:, 1] - .05)
        mask[index] = bool(np.any(hold))
    return y, mask


def audit(directory, output, extras):
    if output.exists():
        raise ValueError('Preserve completed boundary-supervision audit')
    torch.set_num_threads(2)
    cohorts = collect(directory, extras)
    rows = []
    for group, raw in zip(('train', 'validation'), cohorts, strict=True):
        for item in prepare(directory, raw, baseline_evidence=True, full_register=True):
            pairs = boundaries(item)
            y, mask = labels(item['reference'], item['events'], pairs)
            rows.append({'group': group, 'id': item['id'], 'corpus': item['corpus'],
                         'eligible_boundaries': len(pairs), 'genuine_attacks': int(y[mask].sum()),
                         'split_holds': int(np.sum((y == 0) & mask)), 'ambiguous': int(np.sum(~mask))})
    result = {'version': VERSION, 'label_code_sha256': digest(Path(__file__)), 'baseline_hashes': hashes(),
              'extra_manifest_sha256': {str(path): digest(path) for path in extras},
              'items': rows, 'counts': {group: {key: sum(row[key] for row in rows if row['group'] == group)
              for key in ('eligible_boundaries', 'genuine_attacks', 'split_holds', 'ambiguous')}
              for group in ('train', 'validation')},
              'scope': 'Training/validation only; not a fitted model; no test or user audio; no score generation'}
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--extra', type=Path, action='append', default=[])
    args = parser.parse_args()
    audit(args.directory, args.output, args.extra)
