"""Add later piano training/validation passages with frozen annotation clocks.

Training/validation offsets preserve existing non-test performer groups.
Reserved test offsets are prepared only after frozen regression gates pass.
No user recordings, notation generation or new downloads are involved.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from cache_note_verifier import cache_one


def prepare(directory, fresh=False, test_offset=60, training_offset=30):
    if training_offset not in (30, 60, 90) or (fresh and training_offset != 30):
        raise ValueError('Training offsets cannot be applied to reserved test passages')
    if test_offset not in (60, 90, 120) or (not fresh and test_offset != 60):
        raise ValueError('Only reserved test passages may use the later evaluation offset')
    name = 'relational-piano-fresh' if fresh else 'relational-piano-training'
    if fresh and test_offset != 60:
        name += f'-{test_offset}'
    elif not fresh and training_offset != 30:
        name += f'-{training_offset}'
    root = directory / name
    root.mkdir(exist_ok=False)
    clocks = json.loads((directory / 'note-verifier-v3-data/clock-audit.json').read_text())
    items = []
    for clock in clocks:
        if (clock['group'] == 'test') != fresh:
            continue
        if clock['group'] not in ('train', 'validation', 'test'):
            raise ValueError('Unexpected dataset split')
        identity = clock['id'].removeprefix('vienna-')
        sources = list((directory / 'vienna/audio').rglob(identity + '.wav'))
        if len(sources) != 1:
            raise ValueError(f'Missing or ambiguous dataset source: {identity}')
        offset = test_offset if fresh else training_offset
        if sf.info(sources[0]).duration < offset + 5:
            continue
        samples, rate = librosa.load(sources[0], sr=22050, offset=offset, duration=30)
        path = root / (identity + '.wav')
        sf.write(path, samples, rate, subtype='FLOAT')
        midi = pretty_midi.PrettyMIDI(str(directory / 'vienna/midi' / (identity + '.mid')))
        reference = np.asarray([[n.start, n.end, n.pitch] for part in midi.instruments for n in part.notes])
        reference[:, :2] += clock['initial_shift'] + clock['spectral_correction'] - offset
        stop = len(samples) / rate - .25
        reference = reference[(reference[:, 0] >= .25) & (reference[:, 0] < stop)]
        reference[:, 1] = np.minimum(reference[:, 1], len(samples) / rate)
        if not len(reference):
            raise ValueError(f'No labeled notes in eligible source: {identity}')
        items.append({'id': name + '-' + identity, 'corpus': 'vienna-piano',
            'group': clock['group'], 'audio': str(path.resolve()),
            'reference': reference.tolist(), 'evaluation_window': [.25, stop],
            'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True})
    manifest = {'items': items, 'scope': 'Later passages, same compositions; performers stay in their original split',
                'clock': 'Unchanged first-15-second audio calibration; no label fitting from predictions',
                'selection': f'All eligible {"test" if fresh else "training/validation"} recordings at least {offset + 5} seconds long'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Two bounded feature workers; model training waits for this preparation.
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks = [(str(directory / 'note-verifier-context-expanded'), item) for item in items]
        for index, _ in enumerate(pool.map(cache_one, tasks), 1):
            print(json.dumps({'cached': index, 'total': len(items)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--test-offset', type=int, choices=(60, 90, 120), default=60)
    parser.add_argument('--training-offset', type=int, choices=(30, 60, 90), default=30)
    args = parser.parse_args()
    prepare(args.directory, args.fresh, args.test_offset, args.training_offset)
