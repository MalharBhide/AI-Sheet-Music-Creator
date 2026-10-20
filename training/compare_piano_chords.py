"""Compare installed detectors on the chord experiment's development piano data.

No new weights, user audio, reserved works, or production routing changes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.piano_transcription import _PianoEngine
from train_chord_notes import digest, evaluate, write


def compare(root, output, checkpoint):
    manifest = json.loads((output / 'manifest.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    observations_path = root / 'musicnet-v32-observations-v1' / 'manifest.json'
    if (manifest['plan_sha256'] != digest(output / 'plan.json')
            or plan['observations_manifest_sha256'] != digest(observations_path)):
        raise ValueError('Changed chord dataset contract')
    observations = {i['id']: i for i in json.loads(observations_path.read_text())['items']}
    destination = output / 'piano-detector-comparison'
    destination.mkdir(exist_ok=True)
    if (destination / 'report.json').exists():
        raise ValueError('Preserve completed detector comparison')
    run_plan = {'checkpoint_sha256': digest(checkpoint), 'source_plan_sha256': digest(output / 'plan.json'),
                'script_sha256': digest(Path(__file__)), 'threshold': .3,
                'runtime_sha256': digest(Path(__file__).parents[1] / 'backend/app/services/piano_transcription.py'),
                'production_routing_changed': False, 'reserved_works_untouched': True}
    if (destination / 'plan.json').exists() and json.loads((destination / 'plan.json').read_text()) != run_plan:
        raise ValueError('Changed detector experiment')
    write(destination / 'plan.json', run_plan)
    torch.set_num_threads(2)
    engine = _PianoEngine(checkpoint, 'balanced')
    items, predictions = [], []
    for row in manifest['items']:
        if row['group'] != 'validation':
            continue
        source = observations[row['id']]
        audio = Path(source['audio'])
        cache = output / (row['id'] + '.npz')
        if digest(audio) != source['audio_sha256'] or digest(cache) != row['sha256']:
            raise ValueError('Changed development waveform or reference')
        with np.load(cache, allow_pickle=False) as saved:
            item = {**row, 'reference': saved['reference'], 'events': saved['events'], 'released': saved['released']}
        path = destination / (row['id'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['plan_sha256']) != digest(destination / 'plan.json'):
                    raise ValueError('Stale piano detector observation')
                prediction = saved['events']
        else:
            midi = engine.predict(audio, 'piano', 120.)
            prediction = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                for part in midi.instruments for n in part.notes]).reshape(-1, 4)
            np.savez_compressed(path, events=prediction, plan_sha256=digest(destination / 'plan.json'))
        items.append(item)
        predictions.append(prediction)
        print(json.dumps({'observed': len(items), 'id': row['id'], 'notes': len(prediction)}), flush=True)
    report = {'plan': run_plan, 'basic_pitch_raw': evaluate(items, [i['events'] for i in items]),
              'released_accompaniment_verifier': evaluate(items, [i['released'] for i in items]),
              'dedicated_piano_detector': evaluate(items, predictions),
              'limitations': ['Previously used piano development excerpts only, not full-song stems or a test set.',
                  'Basic Pitch accompaniment range is 36-95; dedicated piano range is 21-108.',
                  'These are upstream note events, before arrangement and notation.',
                  'Existing solo-piano mode already uses the dedicated detector; this is not a new model release.']}
    # Like-for-like supported register, alongside full piano-range totals.
    bounded_items = [{**i, 'reference': i['reference'][(i['reference'][:, 2] >= 36) & (i['reference'][:, 2] < 96)]} for i in items]
    bounded_predictions = [p[(p[:, 2] >= 36) & (p[:, 2] < 96)] for p in predictions]
    report['common_register'] = {
        'basic_pitch_raw': evaluate(bounded_items, [i['events'] for i in items]),
        'released_accompaniment_verifier': evaluate(bounded_items, [i['released'] for i in items]),
        'dedicated_piano_detector': evaluate(bounded_items, bounded_predictions)}
    write(destination / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/training'))
    parser.add_argument('--output', type=Path, default=Path('/training/chord-notes-v1'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/opt/models/piano.pth'))
    args = parser.parse_args()
    compare(args.root, args.output, args.checkpoint)
