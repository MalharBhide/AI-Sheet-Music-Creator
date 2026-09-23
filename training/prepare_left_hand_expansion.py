"""Cache unused GuitarSet train/validation tails without touching user uploads.

Use every eligible non-test recording's seconds 30–60. Existing exclusions and
performer groups come from the audited split, never from model accuracy.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from cache_note_verifier import cache_one


def eligible_entries(split):
    for group in ('train', 'validation'):
        for item in split['tracks'][group]:
            if item['corpus'] != 'guitarset':
                continue
            allowed = {'00', '01', '02', '03'} if group == 'train' else {'04'}
            if item['player'] not in allowed:
                raise ValueError('A performer crossed its reserved dataset split')
            yield group, item


def crop_labels(jams, duration):
    return sorted([[n['time'] - 30, min(duration, n['time'] + n['duration'] - 30), float(n['value'])]
                   for annotation in jams['annotations'] if annotation['namespace'] == 'note_midi'
                   for n in annotation['data'] if .25 <= n['time'] - 30 < duration - .25
                   and n['duration'] > 0])


def prepare(directory, workers):
    context = directory / 'note-verifier-context-expanded'
    split_path = context / 'note-verifier-split.json'
    split = json.loads(split_path.read_text())
    root = directory / 'left-hand-expanded-data'
    root.mkdir(exist_ok=True)
    manifest_path = root / 'manifest.json'
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_split_sha256'] != digest:
            raise ValueError('Source split changed; preserve the existing prepared data')
    else:
        items = []
        for group, item in eligible_entries(split):
            source = (context / item['audio']).resolve()
            info = sf.info(source)
            if info.duration < 35:
                continue
            samples, rate = sf.read(source, start=30 * info.samplerate,
                                    stop=60 * info.samplerate, dtype='float32')
            duration = len(samples) / rate
            path = root / (item['id'] + '.wav')
            sf.write(path, samples, rate, subtype='FLOAT')
            labels = list((directory / 'guitarset/annotations').rglob(item['id'] + '.jams'))
            if len(labels) != 1:
                raise ValueError(f'Missing or ambiguous labels: {item["id"]}')
            reference = crop_labels(json.loads(labels[0].read_text()), duration)
            if not reference:
                raise ValueError(f'No labeled notes in eligible passage: {item["id"]}')
            items.append({'id': 'left-expansion-' + item['id'], 'corpus': 'guitarset',
                          'group': group, 'player': item['player'], 'audio': str(path.resolve()),
                          'reference': reference, 'evaluation_window': [.25, duration - .25],
                          'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True})
        manifest = {'items': items, 'source_split_sha256': digest,
                    'selection': 'All train/validation GuitarSet recordings at least 35 seconds long; seconds 30–60',
                    'rights': 'Existing GuitarSet CC BY 4.0 sources; no user uploads',
                    'scope': 'More passages, same performers/compositions, not independent new recordings'}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    tasks = [(str(context), item) for item in manifest['items']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, _ in enumerate(pool.map(cache_one, tasks), 1):
            print(json.dumps({'cached': index, 'total': len(tasks)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    prepare(args.directory, args.workers)
