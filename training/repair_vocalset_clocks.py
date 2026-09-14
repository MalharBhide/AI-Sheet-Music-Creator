"""Migrate pre-audit annotation clocks; preserve split and acoustic features."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cache_vocalset import reconcile_clocks


def repair(directory):
    path = directory / 'vocalset-split.json'
    original = json.loads(path.read_text())
    if original.get('annotation_clock_version') == 1:
        return
    corrected = reconcile_clocks(json.loads(path.read_text()), directory)
    changes = []
    for split, entries in corrected['tracks'].items():
        for old, new in zip(original['tracks'][split], entries, strict=True):
            assert old['id'] == new['id']
            target = directory / 'vocalset-features' / (new['id'] + '.npz')
            with np.load(target) as cache:
                data = {key: cache[key] for key in cache.files}
            old_hash = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
            new_hash = hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()
            if new['annotation_clock_scale'] != 1:
                changes.append({'split': split, 'track': new['id'], 'factor': new['annotation_clock_scale']})
            if str(data['label_hash']) == new_hash:
                continue  # Resume safely after an interrupted migration.
            if str(data['label_hash']) != old_hash:
                raise ValueError('Unexpected original cache identity')
            data['reference'] = np.asarray(new['notes'])
            data['label_hash'] = new_hash
            temporary = target.with_suffix('.npz.tmp')
            with temporary.open('wb') as output:
                np.savez_compressed(output, **data)
            temporary.replace(target)
    (directory / 'vocalset-clock-audit.json').write_text(json.dumps({
        'rule': 'Exact 1x, 0.5x or 2x audio-duration/annotation-duration ratio within 2%; no prediction-based alignment',
        'changes': changes, 'split_unchanged': True, 'test_predictions_inspected': False}, indent=2) + '\n')
    path.write_text(json.dumps(corrected, indent=2) + '\n')
    print(json.dumps(changes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    repair(parser.parse_args().directory)
