"""Select a conservative correction using validation recordings only.

Inputs are locally trained sklearn experiment files, never downloaded pickles.
The runtime export contains only numerical arrays. Full replacements use their
separate, unchanged gate; this experiment preserves the live V2 decisions unless
the context classifier is sufficiently confident to change a shared candidate.
"""

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


def select(directory, model_paths, run_name):
    output = directory / run_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    validation = load_data(directory, 'validation')
    old = torch.load(directory.parent / 'note-verifier-v2-conservative/candidate.pt',
                     weights_only=True, map_location='cpu')
    net = NoteVerifier().eval()
    net.load_state_dict(old['state_dict'])
    for item in validation:
        x, events = previous_events(directory, item)
        with torch.inference_mode():
            probability = torch.sigmoid(net(torch.from_numpy(
                (x - old['mean'].numpy()) / old['scale'].numpy()))).numpy()
        keep = probability >= old['threshold']
        lookup = {tuple(event): index for index, event in enumerate(item['events'])}
        item.update({'old_events': events, 'old_keep': keep,
                     'indices': np.asarray([lookup.get(tuple(event), -1) for event in events]),
                     'before': metrics(item['reference'], events[keep], item['seconds']),
                     'before_matched': matched_references(item['reference'], events[keep])})
    best, search = None, []
    for path in model_paths:
        with path.open('rb') as stream:
            model = pickle.load(stream)
        scores = []
        for item in validation:
            probability = model.predict_proba(item['x'])[:, 1]
            shared = item['indices'] >= 0
            mapped = np.full(len(item['old_events']), np.nan)
            mapped[shared] = probability[item['indices'][shared]]
            scores.append(mapped)
        for prune in (0., .0001, .0003, .001, .003, .005, .01, .02, .03, .05):
            for rescue in (.5, .7, .8, .9, .95, .98, .99, .995, 1.01):
                rows, failed, benefit = [], [], 0
                for item, probability in zip(validation, scores, strict=True):
                    keep = item['old_keep'].copy()
                    keep[probability < prune] = False
                    keep[probability >= rescue] = True
                    after = metrics(item['reference'], item['old_events'][keep], item['seconds'])
                    preserved = item['before_matched'].issubset(
                        matched_references(item['reference'], item['old_events'][keep]))
                    if not preserved or after['false_positives'] > item['before']['false_positives']:
                        failed.append(item['id'])
                    benefit += (item['before']['false_positives'] - after['false_positives']
                                + after['true_positives'] - item['before']['true_positives'])
                    rows.append({'id': item['id'], 'corpus': item['corpus'],
                                 'previous_v2': item['before'], 'trained': after,
                                 'matched_references_preserved': preserved})
                search.append({'model': str(path), 'prune': prune, 'rescue': rescue,
                               'benefit': benefit, 'failed': failed})
                if not failed and benefit > 0 and (best is None or benefit > best[0]):
                    best = (benefit, model, prune, rescue, path, rows)
    write(output / 'search.json', search)
    if best is None:
        write(output / 'selection.json', {'selected': False})
        return
    benefit, model, prune, rescue, path, rows = best
    export(model, prune, output / 'candidate.npz', validation)
    write(output / 'selection.json', {
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.npz').read_bytes()).hexdigest(),
        'prune': prune, 'rescue': rescue, 'source_model': str(path),
        'source_model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'validation_benefit': benefit, 'validation_clips': len(validation),
        'mode': 'V2 correction on shared decoder candidates; every previously matched reference preserved and false positives nonincreasing in every validation recording',
        'test_not_evaluated': True})
    write(output / 'validation.json', {'per_recording': rows})
    print((output / 'selection.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('models', type=Path, nargs='+')
    parser.add_argument('--run', default='correction-validation')
    args = parser.parse_args()
    select(args.directory, args.models, args.run)
