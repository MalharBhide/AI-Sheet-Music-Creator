"""Audit a frozen candidate against the original detector and deployed V2.

All recordings here are consumed regression sets, not a fresh accuracy benchmark.
References use repaired clocks; the previous checkpoint is never retrained.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from cache_note_verifier import cache_one
from note_forest import NoteForest
from note_verifier_model import NoteVerifier
from train_note_verifier import aggregate, frozen, load_data, metrics, probabilities, write
from verifier_release_gate import decision

from app.services.note_context_model import ContextNoteModel


def failures(rows):
    return [row['id'] for row in rows if
            row['trained']['f1'] < row['baseline']['f1']
            or row['trained']['recall'] < row['baseline']['recall'] - .02]


def run(directory, run_name='note-verifier-v3'):
    output = directory / run_name / 'regression.json'
    if output.exists():
        raise ValueError('Preserve the frozen regression result')
    forest_path = directory / run_name / 'candidate.json'
    context_path = directory / run_name / 'candidate.npz'
    consensus = run_name == 'note-verifier-consensus'
    if context_path.exists():
        digest = hashlib.sha256(context_path.read_bytes()).hexdigest()
        if digest != json.loads((directory / run_name / 'selection.json').read_text())['checkpoint_sha256']:
            raise ValueError('Context model changed after validation freeze')
        with np.load(context_path, allow_pickle=False) as arrays:
            model = ContextNoteModel(arrays)
        saved = {'threshold': model.threshold}
    elif consensus:
        model, saved, digest = frozen(directory, 'note-verifier-v3')
        forest_path = directory / 'note-verifier-v4/candidate.json'
        forest_saved = json.loads(forest_path.read_text())
        forest = NoteForest(forest_saved)
        forest_digest = hashlib.sha256(forest_path.read_bytes()).hexdigest()
        if forest_digest != json.loads((directory / 'note-verifier-v4/selection.json').read_text())['checkpoint_sha256']:
            raise ValueError('Forest changed after validation freeze')
        output.parent.mkdir(exist_ok=False)
        write(output.parent / 'selection.json', {
            'neural_sha256': digest, 'forest_sha256': forest_digest,
            'neural_threshold': saved['threshold'], 'forest_threshold': forest_saved['threshold'],
            'selection': 'Reject only with agreement at the two unchanged validation-selected thresholds. Post-failure development decision; regression sets consumed.'})
    elif forest_path.exists():
        digest = hashlib.sha256(forest_path.read_bytes()).hexdigest()
        if digest != json.loads((directory / run_name / 'selection.json').read_text())['checkpoint_sha256']:
            raise ValueError('Forest changed after validation freeze')
        saved = json.loads(forest_path.read_text())
        model = NoteForest(saved)
        with np.load(directory / run_name / 'export-check.npz') as exported:
            np.testing.assert_allclose(model.probability(exported['x']), exported['probability'], atol=1e-12)
    else:
        model, saved, digest = frozen(directory, run_name)
    parent = directory.parent
    previous = torch.load(parent / 'note-verifier-v2-conservative/candidate.pt',
                          map_location='cpu', weights_only=True)
    old_model = NoteVerifier().eval()
    old_model.load_state_dict(previous['state_dict'])
    items = load_data(directory, 'test')
    for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test'):
        manifest = json.loads((parent / name / 'manifest.json').read_text())
        for entry in manifest['items']:
            entry['audio'] = '../' + entry['audio']
            entry['candidate_decoder'] = 'bounded-accompaniment-v1'
            if context_path.exists():
                entry['context_features'] = True
            cache_one((str(directory), entry))
            with np.load(directory / 'note-verifier-features' / (entry['id'] + '.npz')) as cached:
                items.append({**entry, **{k: cached[k] for k in ('x', 'events', 'reference', 'seconds')}})
            print(json.dumps({'cached': entry['id']}), flush=True)
    scores = ([model.probability(item['x']) for item in items] if context_path.exists() or (forest_path.exists() and not consensus)
              else probabilities(model, items, saved['mean'].numpy(), saved['scale'].numpy()))
    if consensus:
        # A retained-note mask, not a calibrated probability.
        scores = [((score >= saved['threshold']) | (forest.probability(item['x']) >= forest_saved['threshold'])).astype(float)
                  for item, score in zip(items, scores, strict=True)]
        saved = {**saved, 'threshold': .5}
    rows = []
    for item, probability in zip(items, scores, strict=True):
        with np.load(parent / 'note-verifier-features' / (item['id'] + '.npz')) as old:
            events, x = old['events'], old['x']
        # V2 deployed a post-MIDI pitch filter; its original cached predictions
        # predate that filter. Also align every system to the same scoring window.
        keep = (events[:, 2] >= 36) & (events[:, 2] < 96)
        if 'evaluation_window' in item:
            start, stop = item['evaluation_window']
            keep &= (events[:, 0] >= start) & (events[:, 0] < stop)
        events, x = events[keep], x[keep]
        with torch.inference_mode():
            old_probability = torch.sigmoid(old_model(torch.from_numpy(
                (x - previous['mean'].numpy()) / previous['scale'].numpy()))).numpy()
        systems = {'baseline': item['events'],
                   'trained': item['events'][probability >= saved['threshold']],
                   'previous_v2': events[old_probability >= previous['threshold']]}
        rows.append({'id': item['id'], 'corpus': item['corpus'],
                     **{key: metrics(item['reference'], value, item['seconds'])
                        for key, value in systems.items()}})
    totals = {corpus: {system: aggregate([row[system] for row in rows if row['corpus'] == corpus])
                       for system in ('baseline', 'trained', 'previous_v2')}
              for corpus in sorted({item['corpus'] for item in items})}
    failed = failures(rows)
    corpus_pass = all(row['trained']['f1'] >= row['baseline']['f1']
                      and row['trained']['precision'] >= row['baseline']['precision']
                      for row in totals.values())
    result = {'checkpoint_sha256': digest, 'threshold': saved['threshold'],
              'status': 'Consumed regression sets; not an independent generalization estimate',
              'passes_original_detector_guard': not failed and corpus_pass, 'failed_recordings': failed,
              'aggregate': totals, 'per_recording': rows}
    result['release'] = decision(result)
    result['passes'] = result['release']['promoted']
    write(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_recording'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='note-verifier-v3')
    args = parser.parse_args()
    run(args.directory, args.run)
