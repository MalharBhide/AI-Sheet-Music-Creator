"""Check frozen/exported CNN parity and unchanged note objects in the website filter."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from app.services import bass_temporal as service
from app.services.temporal_note_model import probability
from bass_temporal_data import load_cached
from bass_temporal_release import load_frozen
from export_bass_temporal import fresh_items
from fresh_bass_temporal import require_regression
from prepare_robust_training_stems import digest, preserve
from prepare_temporal_regression import load_regression


def verify(run, regression, fresh):
    plan, winner, model, normalizer = load_frozen(run)
    require_regression(run, winner)
    prepared = json.loads((Path(plan['sequence_root']) / 'plan.json').read_text())
    items = (load_cached(Path(plan['sequence_root']), ('train', 'validation'))
             + load_regression(Path(prepared['parent_root']), regression, run, winner)
             + fresh_items(run, fresh, winner))
    exported = json.loads((run / 'portable/export.json').read_text())
    if (len(items) != 929 or not exported['passes']
            or exported['sha256'] != {service.ASSET.name: service.EXPECTED_SHA}
            or exported['checkpoint_sha256'] != winner['checkpoint_sha256']):
        raise ValueError('Incomplete or changed exported temporal evidence')
    verifier = service.BassTemporal()
    error, count, rejected = 0., 0, 0
    for item in items:
        x = service.features(item['events'], item['base_x'])
        np.testing.assert_array_equal(x, item['x'], err_msg=item['id'])
        torch.set_num_threads(2)
        expected = probability(model, item['frames'], x, normalizer)
        torch.set_num_threads(4)
        actual = verifier.model.probability(item['frames'], x)
        np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6, err_msg=item['id'])
        error = max(error, float(np.max(np.abs(expected-actual), initial=0.)))
        g = verifier.guardian.probability(item['base_x'])
        keep = ~(item['eligible'] & (expected < winner['threshold']) & (g < winner['guardian_threshold']))
        np.testing.assert_array_equal(keep, ~(item['eligible'] & (actual < .005) & (g < .3)), err_msg=item['id'])
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v))
                 for s, e, p, v in item['events']]
        original = [vars(n).copy() for n in notes]
        midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes.copy()), SimpleNamespace(notes=[])])
        # Runtime receives the frozen original acoustic/CQT caches. Native
        # routing separately checks extraction from real fixture audio.
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(item['duration']*22050), np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=item['base_x']), \
                patch.object(service, 'spectrum', return_value=None), \
                patch.object(service, 'sequences', return_value=item['frames']):
            removed = verifier.filter('cached-dataset-evidence.wav', {}, midi)
        assert torch.get_num_threads() == 4, item['id']
        assert removed == int((~keep).sum()) and not midi.instruments[1].notes, item['id']
        assert [id(n) for n in midi.instruments[0].notes] == [id(n) for n, k in zip(notes, keep, strict=True) if k], item['id']
        assert original == [vars(n) for n in notes], item['id']
        count += len(notes)
        rejected += removed
    report = {'passes': True, 'recordings': len(items), 'v14_retained_events': count,
              'rejected_events': rejected, 'maximum_probability_error': error, 'tolerance': 1e-6,
              'thread_counts': [2, 4], 'same_decisions': True, 'global_threads_unchanged_by_filter': True,
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'arrays': exported['sha256'], 'runtime_source_sha256': digest(Path(service.__file__)),
              'script_sha256': digest(Path(__file__)),
              'scope': 'Cached acoustic/CQT runtime parity across all fitting/validation/consumed/fresh records. Exact retained pitch/start/end/velocity/object/part identity. Rejected total includes fitting, not independent accuracy. No user audio or scores.'}
    preserve(run / 'runtime-parity.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('regression', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    verify(args.run, args.regression, args.fresh)
