"""Verify portable bass merging against frozen sklearn across every cached cohort."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from app.services import bass_articulation as service
from app.services.bass_verifier import BassVerifier
from bass_articulation_labels import annotate
from bass_articulation_release import load_frozen
from bass_boundary_evidence import features, merge
from bass_predictions import probability
from bass_training_data import load
from current_bass_baseline import prepare
from export_bass_articulation import require_report
from prepare_robust_training_stems import digest, preserve


def verify(run, fresh):
    output = run / 'runtime-parity.json'
    if output.exists():
        raise ValueError('Preserve bass articulation runtime witness')
    plan, winner, models = load_frozen(run)
    require_report(run, winner, 'consumed-regression', positive=True)
    first = require_report(run, winner, 'original-first-pass', positive=True)
    if first['test_manifest_sha256'] != digest(fresh / 'manifest.json'):
        raise ValueError('Changed first-pass corpus')
    raw = [{**item, '_root': Path(root)} for root in plan['manifests'] for group in ('train', 'validation')
           for item in load(Path(root), group)]
    raw.extend({**item, '_root': Path(root)} for root in [*plan['consumed_regression_manifests'], str(fresh)]
               for item in load(Path(root), 'test'))
    items = annotate(prepare(raw))
    verifier, baseline = service.BassArticulation(), BassVerifier()
    count, boundaries, merged_count, maximum_error = 0, 0, 0, 0.
    for item in items:
        indices = np.flatnonzero(item['baseline_keep'])
        compact = {**item, 'events': item['events'][indices], 'x': item['x'][indices],
                   'baseline_p': item['baseline_p'][indices], 'baseline_g': item['baseline_g'][indices],
                   'keep': np.ones(len(indices), bool), 'shared': item['shared'][indices]}
        pairs = service.boundaries(compact)
        np.testing.assert_array_equal(indices[pairs], item['pairs'], err_msg=item['id'])
        confidence = []
        for native, portable, acoustic in zip(models, verifier.models, (False, True), strict=True):
            expected_x = features(item, item['pairs'], acoustic=acoustic)
            actual_x = service.features(compact, pairs, acoustic=acoustic)
            np.testing.assert_array_equal(actual_x, expected_x, err_msg=item['id'])
            expected = probability(native, expected_x)
            actual = portable.probability(actual_x)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=item['id'])
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected), initial=0.)))
            confidence.append(expected)
        expected_events, merged = merge(item, item['pairs'], *confidence, winner['threshold'], winner['guardian_threshold'])
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v), origin=int(i))
                 for i, (s, e, p, v) in enumerate(item['events'])]
        attributes = [vars(note).copy() for note in notes]
        owners, expected_keep = np.arange(len(notes)), item['baseline_keep'].copy()
        for before, after in merged:
            first_index, second_index = owners[before], owners[after]
            attributes[first_index]['end'] = max(attributes[first_index]['end'], attributes[second_index]['end'])
            expected_keep[second_index] = False
            owners[owners == second_index] = first_index
        parts = [SimpleNamespace(notes=[notes[i] for i in indices]), SimpleNamespace(notes=[])]
        midi = SimpleNamespace(instruments=parts)
        with np.load(item['_root'] / 'features' / (item['id'] + '.npz'), allow_pickle=False) as saved:
            duration = float(saved['duration'])
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(duration * 22050), np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=compact['x']):
            actual_merges = verifier.filter('cached-evidence.wav', {}, midi, baseline)
        actual_notes = [note for part in parts for note in part.notes]
        assert [id(note) for note in actual_notes] == [id(note) for note, kept in zip(notes, expected_keep, strict=True) if kept], item['id']
        assert not parts[1].notes and actual_merges == len(merged), item['id']
        assert [vars(note) for note in notes] == attributes, item['id']
        actual_events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in actual_notes]).reshape(-1, 4)
        np.testing.assert_array_equal(actual_events, expected_events, err_msg=item['id'])
        count += len(indices)
        boundaries += len(pairs)
        merged_count += len(merged)
    result = {'passes': True, 'recordings': len(items), 'v10_retained_events': count,
              'boundaries': boundaries, 'merged_boundaries': merged_count,
              'maximum_probability_error': maximum_error, 'tolerance': 1e-12,
              'checkpoint_sha256': winner['checkpoint_sha256'], 'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'arrays': {path.name: expected for path, expected, _ in service.CHECKPOINTS},
              'runtime_source_sha256': digest(Path(service.__file__)), 'script_sha256': digest(Path(__file__)),
              'checks': 'Every train/validation/consumed/first-pass cached cohort: exact pair/feature equality, sklearn probabilities, notes/identity/source part and end extensions. All actual bounded Basic Pitch candidates have one source instrument; cross-part merging is separately forbidden by unit test.',
              'scope': 'No audio inference, user uploads or scores; merged total includes training and is not independent accuracy.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    verify(args.run, args.fresh)
