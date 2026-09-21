"""Evaluate a frozen, selective correction of the deployed V2 note decisions."""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import torch
from cache_note_verifier import cache_one
from note_verifier_model import NoteVerifier
from train_note_verifier import aggregate, load_data, metrics, write

from app.services.note_context_model import ContextNoteModel, correction_keep


def matched_references(reference, events):
    return {i for i, _ in mir_eval.transcription.match_notes(
        reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]),
        onset_tolerance=.05, offset_ratio=None)}


def previous_events(directory, item):
    if 'legacy_x' in item:
        x, events = item['legacy_x'], item['legacy_events']
    else:
        with np.load(directory.parent / 'note-verifier-features' / (item['id'] + '.npz')) as cached:
            x, events = cached['x'], cached['events']
    keep = (events[:, 2] >= 36) & (events[:, 2] < 96)
    if 'evaluation_window' in item:
        start, stop = item['evaluation_window']
        keep &= (events[:, 0] >= start) & (events[:, 0] < stop)
    return x[keep], events[keep]


def correction_scores(model, item, events):
    lookup = {tuple(event): score for event, score in zip(
        item['events'], model.probability(item['x']), strict=True)}
    return np.asarray([lookup.get(tuple(event), np.nan) for event in events])


def evaluate(directory, output, items):
    selection = json.loads((output / 'selection.json').read_text())
    expected = selection['checkpoint_sha256']
    checkpoints = ([output / f'candidate-{i}.npz' for i in range(len(expected))]
                   if isinstance(expected, list) else [output / 'candidate.npz'])
    hashes, models = [], []
    for checkpoint in checkpoints:
        hashes.append(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        with np.load(checkpoint, allow_pickle=False) as saved:
            models.append(ContextNoteModel(saved))
    digest = hashes if isinstance(expected, list) else hashes[0]
    if digest != expected:
        raise ValueError('Context model changed after validation freeze')
    previous = torch.load(directory.parent / 'note-verifier-v2-conservative/candidate.pt',
                          map_location='cpu', weights_only=True)
    old_model = NoteVerifier().eval()
    old_model.load_state_dict(previous['state_dict'])
    rows = []
    for item in items:
        x, events = previous_events(directory, item)
        with torch.inference_mode():
            probability = torch.sigmoid(old_model(torch.from_numpy(
                (x - previous['mean'].numpy()) / previous['scale'].numpy()))).numpy()
        before = probability >= previous['threshold']
        scores = np.asarray([correction_scores(model, item, events) for model in models])
        if 'ceiling' in selection:
            if selection['rescue'] <= 1:
                raise ValueError('Agreement correction never restores rejected V2 events')
            shared = np.isfinite(scores).all(axis=0)
            after = correction_keep(probability, np.nan_to_num(scores, nan=1.), shared,
                                    threshold=previous['threshold'], ceiling=selection['ceiling'],
                                    prune=selection['prune'])
        else:
            context = np.maximum.reduce(scores)
            after = before.copy()
            after[context < selection['prune']] = False
            after[context >= selection['rescue']] = True
        old_metrics = metrics(item['reference'], events[before], item['seconds'])
        new_metrics = metrics(item['reference'], events[after], item['seconds'])
        preserved = matched_references(item['reference'], events[before]).issubset(
            matched_references(item['reference'], events[after]))
        rows.append({'id': item['id'], 'corpus': item['corpus'],
                     'previous_v2': old_metrics, 'trained': new_metrics,
                     'matched_references_preserved': preserved,
                     'passes': preserved and new_metrics['false_positives'] <= old_metrics['false_positives']})
    totals = {corpus: {system: aggregate([row[system] for row in rows if row['corpus'] == corpus])
                      for system in ('previous_v2', 'trained')}
              for corpus in sorted({item['corpus'] for item in items})}
    return {'checkpoint_sha256': digest, 'prune': selection['prune'], 'rescue': selection['rescue'],
            'passes': all(row['passes'] for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']],
            'aggregate': totals, 'per_recording': rows}


def external(directory, path, audio_base):
    manifest = json.loads(path.read_text())
    for entry in manifest['items']:
        entry = {**entry, 'audio': str((audio_base / entry['audio']).resolve()),
                 'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True}
        cache_one((str(directory), entry))
        with np.load(directory / 'note-verifier-features' / (entry['id'] + '.npz')) as cached:
            keys = ('x', 'events', 'reference', 'seconds', 'legacy_events', 'legacy_x')
            yield {**entry, **{key: cached[key] for key in keys if key in cached}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='correction-validation')
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--report', choices=['regression.json', 'fresh.json', 'oxford-final.json'])
    args = parser.parse_args()
    output = args.directory / args.run
    result_path = output / (args.report or ('fresh.json' if args.fresh else 'regression.json'))
    if result_path.exists():
        raise ValueError('Preserve the frozen evaluation result')
    torch.set_num_threads(2)
    if args.fresh:
        items = list(external(args.directory, args.fresh, args.directory))
    else:
        items = load_data(args.directory, 'test')
        for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test'):
            items.extend(external(args.directory, args.directory.parent / name / 'manifest.json', args.directory.parent))
    result = evaluate(args.directory, output, items)
    result['status'] = ('Previously unscored later windows, same performers/compositions'
                        if args.fresh else 'Consumed regression recordings')
    write(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)
