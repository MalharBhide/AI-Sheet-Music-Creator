"""Prepare consumed-test envelopes only for a frozen validation-selected CNN."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.temporal_note_model import sequences, spectrum
from bass_temporal_data import baseline_items
from bass_training_data import identity
from prepare_robust_training_stems import digest, preserve


def prepare(run, output):
    from bass_temporal_release import load_frozen

    plan, winner, _, _ = load_frozen(run)
    prepared = json.loads((Path(plan['sequence_root']) / 'plan.json').read_text())
    parent = Path(prepared['parent_root'])
    output.mkdir(exist_ok=False)
    binding = {'checkpoint_sha256': winner['checkpoint_sha256'],
               'batch_selection_sha256': digest(run / 'batch-selection.json'),
               'parent_manifest_sha256': digest(parent / 'manifest.json'),
               'producer_sha256': digest(Path(__file__)), 'no_fitting_or_selection': True}
    preserve(output / 'plan.json', binding)
    work = output / 'features'
    work.mkdir()
    records = []
    for item in baseline_items(parent, ('test',)):
        if digest(Path(item['audio'])) != item['audio_sha256']:
            raise ValueError('Changed consumed audio')
        samples, rate = sf.read(item['audio'], dtype='float32')
        if len(samples) / rate != item['duration']:
            raise ValueError('Changed consumed audio clock')
        frames = sequences(spectrum(samples, rate), item['events'])
        row = {'id': item['id'], 'group': 'test', 'source_group': item['source_group'],
               'source_cache_sha256': item['cache_sha256'], 'audio_sha256': item['audio_sha256']}
        path = work / (row['id'] + '.npz')
        np.savez_compressed(path, frames=frames, events=item['events'], identity=identity(row),
                            plan_sha256=digest(output / 'plan.json'))
        row['cache_sha256'] = digest(path)
        records.append(row)
    if len(records) != 220:
        raise ValueError('Incomplete consumed temporal preparation')
    preserve(output / 'manifest.json', {'plan_sha256': digest(output / 'plan.json'), 'items': records,
             'no_fitting_or_selection': True})
    print(json.dumps({'completed': len(records)}), flush=True)


def load_regression(parent, directory, run, winner):
    manifest = json.loads((directory / 'manifest.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if (manifest['plan_sha256'] != digest(directory / 'plan.json') or not manifest['no_fitting_or_selection']
            or plan['checkpoint_sha256'] != winner['checkpoint_sha256']
            or plan['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or plan['parent_manifest_sha256'] != digest(parent / 'manifest.json')
            or plan['producer_sha256'] != digest(Path(__file__))):
        raise ValueError('Changed consumed temporal bindings')
    originals = {i['id']: i for i in baseline_items(parent, ('test',))}
    result = []
    for row in manifest['items']:
        item = originals.pop(row['id'])
        path = directory / 'features' / (row['id'] + '.npz')
        if (row['group'] != 'test' or row['source_group'] != item['source_group']
                or row['source_cache_sha256'] != item['cache_sha256'] or digest(path) != row['cache_sha256']):
            raise ValueError('Changed consumed temporal source')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['identity']) != identity({k: v for k, v in row.items() if k != 'cache_sha256'})
                    or str(saved['plan_sha256']) != manifest['plan_sha256']):
                raise ValueError('Changed consumed temporal cache identity')
            np.testing.assert_array_equal(saved['events'], item['events'])
            result.append({**item, 'frames': saved['frames']})
    if originals:
        raise ValueError('Incomplete consumed temporal cache')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.run, args.output)
