"""Reserve unused labeled GuitarSet tails; never touch user recordings.

Earlier experiments used only the first 30 seconds. These later sections share
the existing test player and pieces, so they are not an independent performer
or composition benchmark. No test audio/labels are used to fit the classifier.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from cache_note_verifier import cache_one


def prepare(directory):
    root = directory / 'residual-guitar-tails'
    root.mkdir(exist_ok=False)
    context = directory / 'note-verifier-context-expanded'
    manifest = json.loads((context / 'note-verifier-split.json').read_text())
    # Fixed duration rule and all eligible test tracks, with no accuracy selection.
    items = []
    for item in manifest['tracks']['test']:
        if item['corpus'] != 'guitarset':
            continue
        if item['player'] != '05':
            raise ValueError('Expected only the held-out test performer')
        source = (context / item['audio']).resolve()
        info = sf.info(source)
        if info.duration < 35:
            continue
        samples, rate = sf.read(source, start=30 * info.samplerate,
                                stop=60 * info.samplerate, dtype='float32')
        duration = len(samples) / rate
        path = root / (item['id'] + '.wav')
        sf.write(path, samples, rate, subtype='FLOAT')
        annotations = list((directory / 'guitarset/annotations').rglob(item['id'] + '.jams'))
        if len(annotations) != 1:
            raise ValueError(f'Missing or ambiguous dataset labels: {item["id"]}')
        jams = json.loads(annotations[0].read_text())
        reference = [[n['time'] - 30, min(duration, n['time'] + n['duration'] - 30), float(n['value'])]
                     for annotation in jams['annotations'] if annotation['namespace'] == 'note_midi'
                     for n in annotation['data'] if .25 <= n['time'] - 30 < duration - .25
                     and n['duration'] > 0]
        if not reference:
            raise ValueError(f'No reference notes in eligible tail: {item["id"]}')
        entry = {'id': 'residual-tail-' + item['id'], 'corpus': 'guitarset-later',
                 'audio': str(path.resolve()), 'reference': sorted(reference),
                 'evaluation_window': [.25, duration - .25],
                 'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True}
        items.append(entry)
    if not items:
        raise ValueError('No unused evaluation tails available')
    (root / 'manifest.json').write_text(json.dumps({'items': items,
        'scope': 'Previously unused seconds 30–60, existing test performer/pieces',
        'selection': 'Every held-out player 05 recording at least 35 seconds long',
        'rights': 'Pinned GuitarSet CC BY 4.0 dataset; no user uploads'}, indent=2) + '\n')
    for index, item in enumerate(items, 1):
        cache_one((str(context), item))
        print(json.dumps({'cached': index, 'total': len(items)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
