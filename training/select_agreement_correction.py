"""Freeze a two-head, uncertainty-limited correction using validation only."""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from evaluate_context_correction import matched_references, previous_events
from export_context_verifier import export
from note_verifier_model import NoteVerifier
from train_note_verifier import load_data, metrics, write

from app.services.note_context_model import correction_keep, shared_events


def select(directory, paths, run_name):
    if len(paths) != 2:
        raise ValueError('Exactly two independently fitted context heads are required')
    output = directory / run_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    # These are local training outputs, never untrusted downloaded pickles.
    models = []
    for path in paths:
        with path.open('rb') as stream:
            models.append(pickle.load(stream))
    old = torch.load(directory.parent / 'note-verifier-v2-conservative/candidate.pt',
                     weights_only=True, map_location='cpu')
    net = NoteVerifier().eval()
    net.load_state_dict(old['state_dict'])
    validation = load_data(directory, 'validation')
    for item in validation:
        x, events = previous_events(directory, item)
        with torch.inference_mode():
            probability = torch.sigmoid(net(torch.from_numpy(
                (x - old['mean'].numpy()) / old['scale'].numpy()))).numpy()
        keep = probability >= old['threshold']
        lookup = {tuple(event): index for index, event in enumerate(item['events'])}
        shared = shared_events(events, item['events'])
        context = np.ones((2, len(events)))
        for index, model in enumerate(models):
            scores = model.predict_proba(item['x'])[:, 1]
            context[index, shared] = [scores[lookup[tuple(event)]] for event in events[shared]]
        item.update({'events_old': events, 'p': probability, 'context': context, 'shared': shared,
                     'before': metrics(item['reference'], events[keep], item['seconds']),
                     'matched': matched_references(item['reference'], events[keep])})
    best, search = None, []
    for ceiling in (.15, .2):
        for prune in (.0001, .0003, .001, .003, .005, .01, .02, .03, .05):
            rows, failed, benefit = [], [], 0
            for item in validation:
                keep = correction_keep(item['p'], item['context'], item['shared'],
                                       threshold=old['threshold'], ceiling=ceiling, prune=prune)
                events = item['events_old'][keep]
                after = metrics(item['reference'], events, item['seconds'])
                preserved = item['matched'].issubset(matched_references(item['reference'], events))
                if not preserved or after['false_positives'] > item['before']['false_positives']:
                    failed.append(item['id'])
                benefit += item['before']['false_positives'] - after['false_positives']
                rows.append({'id': item['id'], 'corpus': item['corpus'],
                             'previous_v2': item['before'], 'trained': after,
                             'matched_references_preserved': preserved})
            search.append({'ceiling': ceiling, 'prune': prune, 'benefit': benefit, 'failed': failed})
            if not failed and benefit > 0 and (best is None or benefit > best[0]):
                best = (benefit, ceiling, prune, rows)
    write(output / 'search.json', search)
    if best is None:
        write(output / 'selection.json', {'selected': False})
        return
    benefit, ceiling, prune, rows = best
    hashes = []
    for index, model in enumerate(models):
        path = output / f'candidate-{index}.npz'
        export(model, prune, path, validation)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    write(output / 'validation.json', {'per_recording': rows})
    write(output / 'selection.json', {'checkpoint_sha256': hashes, 'prune': prune,
        'ceiling': ceiling, 'rescue': 1.01, 'validation_benefit': benefit, 'test_not_evaluated': True,
        'source_models': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'mode': 'Both context heads must reject an uncertain V2 event; exact shared decoder events only'})
    print((output / 'selection.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('models', type=Path, nargs=2)
    parser.add_argument('--run', default='agreement-correction')
    args = parser.parse_args()
    select(args.directory, args.models, args.run)
