"""Measure which unmatched validation notes a proposed residual policy can reach."""

import argparse
import json
from pathlib import Path

import torch
from cache_note_verifier import targets
from export_left_consensus_v8 import validation_items
from train_left_consensus_v8 import prepare
from train_note_verifier import write


def coverage(items):
    counts = {key: {'matched': 0, 'unmatched': 0} for key in ('eligible', 'protected')}
    for item in items:
        retained = item['keep']
        labels, _ = targets(item['reference'], item['events'][retained])
        eligible = (item['shared'] & (item['p'] <= .5))[retained]
        for group, mask in [('eligible', eligible), ('protected', ~eligible)]:
            counts[group]['matched'] += int(labels[mask].sum())
            counts[group]['unmatched'] += int((1 - labels[mask]).sum())
    return counts


def run(directory, source, output):
    if output.exists():
        raise ValueError('Preserve completed coverage audits')
    torch.set_num_threads(2)
    items = validation_items(directory, source)
    if any(item['group'] != 'validation' for item in items):
        raise ValueError('Coverage diagnosis must not consume test labels')
    prepared = prepare(directory, items, full_register=True)
    result = {'clips': len(items), 'baseline': 'Pinned V7', 'validation_only': True,
              'counts': coverage(prepared),
              'scope': 'All-register residual policy; protected combines strong, nonshared, and window-edge notes; no test or user audio processing'}
    write(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.directory, args.source, args.output)
