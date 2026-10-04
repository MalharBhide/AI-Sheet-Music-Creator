"""Verify frozen repeat-boundary runtime decisions, timing and source identity."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from app.services import accompaniment_verifier as service
from current_accompaniment_baseline import prepare
from repeat_boundary_features import merge
from repeat_boundary_release import load_frozen, require_report
from train_broad_consensus import digest, validation_items
from train_left_consensus_v8 import regression_items
from train_left_relations_v7 import complete_evidence
from train_note_verifier import write
from train_repeat_boundaries import annotate, predict
from train_residual_verifier import cached_external


def verify(directory, run, fresh, current_piano, output):
    if output.exists():
        raise ValueError('Preserve the runtime parity report')
    selection, models = load_frozen(run)
    require_report(run, selection, 'regression', positive=True)
    require_report(run, selection, 'current-piano-regression')
    require_report(run, selection, 'fresh-stress')
    items = validation_items(directory, run) + regression_items(directory)
    items.extend(cached_external(directory, current_piano, directory))
    items.extend(cached_external(directory, fresh, directory))
    torch.set_num_threads(2)
    verifier = service.AccompanimentVerifier()
    assert verifier.boundary_model.threshold == selection['threshold']
    assert verifier.boundary_guardian_model.threshold == selection['guardian_threshold']
    prepared = annotate(prepare(directory, items))
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
        repeat, guardian = predict(models, [item])[0]
        expected_events, merged = merge(item, item['pairs'], repeat, guardian,
                                        selection['threshold'], selection['guardian_threshold'])
        expected = item['keep'].copy()
        owners = np.arange(len(item['events']))
        for before, after in merged:
            first, second = owners[before], owners[after]
            attributes[indices[first]]['end'] = max(attributes[indices[first]]['end'], attributes[indices[second]]['end'])
            expected[second] = False
            owners[owners == second] = first
        duration = min(30., sf.info(directory / item['audio']).duration)
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(duration * 22050), dtype=np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=features), \
                patch.object(service, 'decode_candidates', return_value=bounded):
            verifier.filter('cached-features-only.wav', {}, midi)
        kept_ids = {id(note) for part in parts for note in part.notes}
        actual = np.asarray([id(notes[index]) in kept_ids for index in indices])
        np.testing.assert_array_equal(actual, expected, err_msg=item['id'])
        actual_events = np.asarray([[notes[i].start, notes[i].end, notes[i].pitch, notes[i].velocity]
                                    for i in indices if id(notes[i]) in kept_ids]).reshape(-1, 4)
        np.testing.assert_array_equal(actual_events, expected_events, err_msg=item['id'])
        for original, part in zip(original_parts, parts, strict=True):
            assert [id(n) for n in part.notes] == [id(n) for n in original if id(n) in kept_ids], item['id']
        assert [vars(note) for note in notes] == attributes, item['id']
        total += len(indices)
        rejected += int((~expected).sum())
        additional += int((item['baseline_keep'] & ~expected).sum())
    result = {'passes': True, 'recordings': len(items), 'scored_candidate_events': total,
              'total_scored_rejected': rejected, 'additional_merged_vs_v8': additional,
              'model_sha256': {'relations': service.BOUNDARY_SHA256, 'guardian': service.BOUNDARY_GUARDIAN_SHA256},
              'checkpoint_sha256': selection['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'fresh_manifest_sha256': digest(fresh),
              'current_piano_manifest_sha256': digest(current_piano),
              'checks': 'Runtime matches frozen sklearn decisions with complete original neighboring geometry and V2 confidence; retained identity, source part, pitch, onset and velocity preserved; only selected touching holds extended',
              'scope': 'Scored targets from cached labeled datasets; outside-crop neighbors supply context only. No audio inference or score generation'}
    write(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    parser.add_argument('current_piano', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run, args.fresh, args.current_piano, args.output)
