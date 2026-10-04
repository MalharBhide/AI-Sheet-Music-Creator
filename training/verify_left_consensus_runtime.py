"""Check frozen V7 decisions with complete candidate neighborhoods, without audio inference."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from app.services import accompaniment_verifier as service
from train_left_relations_v7 import complete_evidence, hashes, predict, prepare
from train_note_verifier import load_data, write
from train_residual_verifier import cached_external, residual_keep


def verify(directory, run, output):
    if output.exists():
        raise ValueError('Preserve the runtime parity report')
    selection = json.loads((run / 'selection.json').read_text())
    checkpoint = run / 'candidate.pickle'
    if (not selection['selected'] or selection['baseline_hashes'] != hashes()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']):
        raise ValueError('Frozen candidate changed')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)  # Hash-verified local research artifact only.
    items = load_data(directory, 'validation') + load_data(directory, 'test')
    for name in ('relational-piano-training', 'left-hand-expanded-data'):
        items.extend(item for item in cached_external(directory, directory.parent / name / 'manifest.json', directory)
                     if item['group'] == 'validation')
    for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                 'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails',
                 'relational-piano-fresh', 'relational-piano-fresh-90'):
        base = directory.parent if name.startswith('note-verifier-') else directory
        items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    verifier = service.AccompanimentVerifier()
    assert verifier.left_relations_model.threshold == selection['threshold']
    assert verifier.left_guardian_model.threshold == selection['guardian_threshold']
    prepared = prepare(directory, items)
    total, rejected, additional = 0, 0, 0
    for raw, item in zip(items, prepared, strict=True):
        events, original_x = complete_evidence(directory, item)
        bounds = (events[:, 2] >= 36) & (events[:, 2] < 96)
        events, original_x = events[bounds], original_x[bounds]
        lookup = {tuple(event): index for index, event in enumerate(events)}
        indices = np.asarray([lookup[tuple(event)] for event in item['events']], dtype=int)
        features = np.zeros((len(events), 52), dtype=np.float32)
        features[indices] = item['x'][:, :52]
        features[:, :26] = original_x
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v))
                 for s, e, p, v in events]
        attributes = [vars(note).copy() for note in notes]
        split = len(notes) // 2
        original_parts = [notes[:split], notes[split:]]
        parts = [SimpleNamespace(notes=part.copy()) for part in original_parts]
        midi = SimpleNamespace(instruments=parts)
        shared_keys = {tuple(event) for event in raw['events']}
        bounded = SimpleNamespace(instruments=[SimpleNamespace(notes=[note for note, event in zip(notes, events, strict=True)
                    if tuple(event) in shared_keys])])
        probabilities = predict(models, item['x'], threshold=selection['threshold'],
                                guardian_threshold=selection['guardian_threshold'])
        expected = residual_keep(item, probabilities, selection['threshold'])
        duration = min(30., sf.info(directory / item['audio']).duration)
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(duration * 22050), dtype=np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=features), \
                patch.object(service, 'decode_candidates', return_value=bounded):
            verifier.filter('cached-features-only.wav', {}, midi)
        kept_ids = {id(note) for part in parts for note in part.notes}
        actual = np.asarray([id(notes[index]) in kept_ids for index in indices])
        np.testing.assert_array_equal(actual, expected, err_msg=item['id'])
        for original, part in zip(original_parts, parts, strict=True):
            assert [id(n) for n in part.notes] == [id(n) for n in original if id(n) in kept_ids], item['id']
        assert [vars(note) for note in notes] == attributes, item['id']
        total += len(indices)
        rejected += int((~expected).sum())
        additional += int((item['baseline_keep'] & ~expected).sum())
    result = {'passes': True, 'recordings': len(items), 'scored_candidate_events': total,
              'total_scored_rejected': rejected, 'additional_rejected_vs_v6': additional,
              'model_sha256': {'relations': service.LEFT_RELATIONS_SHA256, 'guardian': service.LEFT_GUARDIAN_SHA256},
              'checks': 'Runtime matches frozen sklearn decisions with complete original neighboring geometry and V2 confidence; retained identity, source part and attributes preserved',
              'scope': 'Scored targets from cached labeled datasets; outside-crop neighbors supply context only. No audio inference or score generation'}
    write(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run, args.output)
