"""Verify safe portable heads and runtime note identities against frozen sklearn."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from app.services import bass_verifier as service
from bass_positive_release import load_frozen, require_report
from bass_predictions import probability
from bass_training_data import load
from prepare_robust_training_stems import digest, preserve


def verify(directory, run, held, reserved):
    output = run / 'runtime-parity.json'
    if output.exists():
        raise ValueError('Preserve consumed runtime verification')
    winner, models = load_frozen(directory, run)
    require_report(run, winner, 'held-regression')
    require_report(run, winner, 'reserved-slakh', positive=True)
    plan = json.loads((run / 'plan.json').read_text())
    data = [directory, *map(Path, plan['additional_manifests'])]
    items = [{**item, '_cache_directory': root} for root in data for group in ('train', 'validation')
             for item in load(root, group)]
    for root in (held, reserved):
        items.extend({**item, '_cache_directory': root} for item in load(root, 'test'))
    verifier = service.BassVerifier()
    count, removed, maximum_error = 0, 0, 0.
    for item in items:
        x = item['x']
        confidence = [probability(model, x[:, :model.n_features_in_]) for model in models]
        for native, portable, expected in zip(models, verifier.models, confidence, strict=True):
            actual = portable.probability(x[:, :native.n_features_in_])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=item['id'])
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected), initial=0.)))
        expected = ~(item['eligible'] & (confidence[0] < winner['threshold']) & (confidence[1] < winner['guardian_threshold']))
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v), origin=item['id'])
                 for s, e, p, v in item['events']]
        attributes = [vars(note).copy() for note in notes]
        split = len(notes) // 2
        originals = [notes[:split], notes[split:]]
        parts = [SimpleNamespace(notes=part.copy()) for part in originals]
        with np.load(item['_cache_directory'] / 'features' / (item['id'] + '.npz'), allow_pickle=False) as saved:
            duration = float(saved['duration'])
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(duration * 22050), dtype=np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=x):
            result = verifier.filter('cached-features.wav', {}, SimpleNamespace(instruments=parts))
        actual_ids = [id(note) for part in parts for note in part.notes]
        expected_ids = [id(note) for note, keep in zip(notes, expected, strict=True) if keep]
        assert actual_ids == expected_ids, item['id']
        for part, original in zip(parts, originals, strict=True):
            assert [id(note) for note in part.notes] == [id(note) for note in original if id(note) in actual_ids], item['id']
        assert attributes == [vars(note) for note in notes], item['id']
        assert result == int((~expected).sum()), item['id']
        count += len(notes)
        removed += result
    result = {'passes': True, 'recordings': len(items), 'scored_candidates': count, 'removed': removed,
              'maximum_probability_error': maximum_error, 'tolerance': 1e-12,
              'checkpoint_sha256': winner['checkpoint_sha256'], 'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'runtime_source_sha256': digest(Path(service.__file__)), 'gate_source_sha256': digest(Path(__file__)),
              'arrays': {path.name: expected for path, expected, _ in service.CHECKPOINTS},
              'checks': 'Every cached cohort: sklearn/portable probabilities, final keep mask, note identity, original source part and all attributes. No audio inference or scores.'}
    preserve(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'run', 'held', 'reserved'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.directory, args.run, args.held, args.reserved)
