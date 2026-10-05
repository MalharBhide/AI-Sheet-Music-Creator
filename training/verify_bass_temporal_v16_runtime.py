"""V16 portable/filter parity on every sealed V15 fitting/regression survivor."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from app.services import bass_temporal as inherited
from app.services import bass_temporal_refinement as service
from app.services.bass_harmonic import features
from app.services.temporal_note_model import probability
from bass_temporal_v16_evidence import all_items
from prepare_robust_training_stems import digest, preserve


def verify(run, fresh):
    _, winner, model, normalizer, items = all_items(run, fresh)
    exported = json.loads((run / 'portable/export.json').read_text())
    if (not exported['passes'] or exported['recordings'] != 1029
            or exported['sha256'] != {service.ASSET.name: service.EXPECTED_SHA}
            or exported['checkpoint_sha256'] != winner['checkpoint_sha256']
            or digest(service.ASSET) != service.EXPECTED_SHA):
        raise ValueError('Incomplete or changed exported V16 evidence')
    verifier = service.BassTemporalRefinement()
    error, count, rejected = 0., 0, 0
    for item in items:
        x = features(item['events'], item['base_x'])
        np.testing.assert_array_equal(x, item['x'], err_msg=item['id'])
        torch.set_num_threads(2)
        expected = probability(model, item['frames'], x, normalizer)
        torch.set_num_threads(4)
        actual = verifier.model.probability(item['frames'], x)
        np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6, err_msg=item['id'])
        error = max(error, float(np.max(np.abs(expected-actual), initial=0.)))
        g = verifier.guardian.probability(item['base_x'])
        keep = ~(item['eligible'] & (expected < winner['threshold']) & (g < winner['guardian_threshold']))
        np.testing.assert_array_equal(keep, ~(item['eligible'] & (actual < verifier.model.threshold)
                                             & (g < verifier.guardian.threshold)), err_msg=item['id'])
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v))
                 for s, e, p, v in item['events']]
        original = [vars(n).copy() for n in notes]
        # Check native single-part ordering first, then cross-part identity.
        for parts, order in (([notes], list(range(len(notes)))),
                             ([notes[::2], notes[1::2]],
                              list(range(0, len(notes), 2)) + list(range(1, len(notes), 2)))):
            ordered = {key: item[key][order] for key in ('base_x', 'frames')}
            ordered_x = features(item['events'][order], ordered['base_x'])
            ordered_expected = probability(model, ordered['frames'], ordered_x, normalizer)
            ordered_g = verifier.guardian.probability(ordered['base_x'])
            ordered_keep = ~(item['eligible'][order] & (ordered_expected < verifier.model.threshold)
                             & (ordered_g < verifier.guardian.threshold))
            if len(parts) == 1:
                np.testing.assert_array_equal(ordered_keep, keep, err_msg=item['id'])
            midi = SimpleNamespace(instruments=[SimpleNamespace(notes=part.copy()) for part in parts])
            with patch.object(inherited.sf, 'read', return_value=(np.zeros(round(item['duration']*22050), np.float32), 22050)), \
                    patch.object(inherited, 'note_features', return_value=ordered['base_x']), \
                    patch.object(inherited, 'spectrum', return_value=None), \
                    patch.object(inherited, 'sequences', return_value=ordered['frames']):
                removed = verifier.filter('cached-dataset-evidence.wav', {}, midi)
            assert torch.get_num_threads() == 4, item['id']
            assert removed == int((~ordered_keep).sum()), item['id']
            kept_ids = {id(notes[i]) for i, k in zip(order, ordered_keep, strict=True) if k}
            for part, prior in zip(midi.instruments, parts, strict=True):
                assert [id(n) for n in part.notes] == [id(n) for n in prior if id(n) in kept_ids], item['id']
            assert original == [vars(n) for n in notes], item['id']
        count += len(notes)
        rejected += int((~keep).sum())
    report = {'passes': True, 'recordings': len(items), 'v15_retained_events': count,
              'rejected_events': rejected, 'maximum_probability_error': error, 'tolerance': 1e-6,
              'thread_counts': [2, 4], 'same_decisions': True, 'global_threads_unchanged_by_filter': True,
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'arrays': exported['sha256'], 'runtime_source_sha256': digest(Path(service.__file__)),
              'inherited_filter_sha256': digest(Path(inherited.__file__)), 'script_sha256': digest(Path(__file__)),
              'scope': 'Every fitting/validation/consumed/fresh input. Frozen Torch/NPZ probabilities, exact retained pitch/start/end/velocity/object/part identity, and no process-global thread mutation. No user audio or scores.'}
    preserve(run / 'runtime-parity.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    verify(args.run, args.fresh)
