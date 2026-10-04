"""Verify broad pitch-preserving runtime on frozen caches without audio inference."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from app.services import accompaniment_verifier as service
from app.services.note_context_model import residual_keep
from export_broad_consensus import release_reports
from train_broad_consensus import digest, load_winner, validation_items
from train_left_consensus_v8 import prepare, probabilities, regression_items
from train_left_relations_v7 import complete_evidence
from train_note_verifier import write
from train_residual_verifier import cached_external


def verify(directory, run, fresh, output):
    if output.exists():
        raise ValueError('Preserve the runtime parity report')
    selection, models = load_winner(run)
    release_reports(run, selection)
    items = validation_items(directory, run) + regression_items(directory)
    items.extend(cached_external(directory, fresh, directory))
    torch.set_num_threads(2)
    verifier = service.AccompanimentVerifier()
    assert verifier.broad_model.threshold == selection['threshold']
    assert verifier.broad_guardian_model.threshold == selection['guardian_threshold']
    prepared = prepare(directory, items, baseline_evidence=True, full_register=True)
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
        probability = probabilities(models, item['x'], selection['threshold'], selection['guardian_threshold'])
        expected = residual_keep(item['keep'], item['p'], item['shared'], probability,
                                 threshold=selection['threshold'], ceiling=1.)
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
              'total_scored_rejected': rejected, 'additional_rejected_vs_v7': additional,
              'model_sha256': {'relations': service.BROAD_SHA256, 'guardian': service.BROAD_GUARDIAN_SHA256},
              'checkpoint_sha256': selection['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'fresh_manifest_sha256': digest(fresh),
              'checks': 'Runtime matches frozen sklearn decisions with complete original neighboring geometry and V2 confidence; retained identity, source part and attributes preserved',
              'scope': 'Scored targets from cached labeled datasets; outside-crop neighbors supply context only. No audio inference or score generation'}
    write(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run, args.fresh, args.output)
