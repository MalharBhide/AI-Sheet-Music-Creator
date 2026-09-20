"""Evaluate the frozen verifier on a predeclared external manifest, once."""

import argparse
import json
from pathlib import Path

import numpy as np
from cache_note_verifier import cache_one
from train_note_verifier import evaluate, frozen, passes_guard, probabilities, write


def run(directory, manifest_path, name, run_name='note-verifier-v1'):
    target = directory / run_name / (name + '.json')
    if target.exists():
        raise ValueError('This evaluation was consumed already')
    model, checkpoint, digest = frozen(directory, run_name)
    manifest = json.loads(manifest_path.read_text())
    items = []
    for item in manifest['items']:
        cache_one((str(directory), item))
        with np.load(directory / 'note-verifier-features' / (item['id'] + '.npz')) as data:
            record = {**item, **{k: data[k] for k in ('x', 'events', 'reference', 'seconds')}}
        if 'evaluation_window' in item:
            start, stop = item['evaluation_window']
            mask = (record['events'][:, 0] >= start) & (record['events'][:, 0] < stop)
            record['x'], record['events'] = record['x'][mask], record['events'][mask]
            record['seconds'] = stop - start
        items.append(record)
        print(json.dumps({'cached': item['id']}), flush=True)
    result = evaluate(items, probabilities(model, items, checkpoint['mean'].numpy(), checkpoint['scale'].numpy()), checkpoint['threshold'])
    result.update({'checkpoint_sha256': digest, 'manifest': manifest, 'passes': passes_guard(result)})
    write(target, result)
    print(json.dumps({'aggregate': result['aggregate'], 'passes': result['passes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('name')
    parser.add_argument('--run', default='note-verifier-v1')
    args = parser.parse_args()
    run(args.directory, args.manifest, args.name, args.run)
