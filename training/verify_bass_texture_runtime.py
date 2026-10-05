"""Exact website texture features, probabilities and note identities on all cohorts."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from app.services import bass_texture as service
from bass_texture_v12_data import load_cached
from bass_texture_v12_release import load_frozen
from export_bass_texture import fresh_items
from fresh_bass_texture_v12 import require_regression
from prepare_robust_training_stems import digest, preserve
from train_bass_texture_v12 import predictions


def verify(run, fresh):
    destination = run / 'runtime-parity.json'
    if destination.exists():
        raise ValueError('Preserve texture runtime witness')
    plan, winner, models = load_frozen(run)
    require_regression(run, winner)
    new_items = fresh_items(run, fresh, winner)
    export = json.loads((run / 'portable/export.json').read_text())
    if (not export['passes'] or export['checkpoint_sha256'] != winner['checkpoint_sha256']
            or export['sha256'] != {path.name: expected for path, expected, _, _ in service.CHECKPOINTS}):
        raise ValueError('Changed texture portable export')
    items = load_cached(Path(plan['baseline_cache_root']), ('train', 'validation', 'test')) + new_items
    verifier = service.BassTexture()
    count, rejected, error = 0, 0, 0.
    for item in items:
        base_x, events = item['x'][:, :52], item['events']
        actual_x = service.features(events, base_x)
        np.testing.assert_array_equal(actual_x, item['x'], err_msg=item['id'])
        expected = [p[0] for p in predictions(models, [item])]
        actual = [model.probability(view) for model, view in zip(verifier.models,
                    (actual_x, service.acoustic_view(actual_x)), strict=True)]
        for p, g in zip(expected, actual, strict=True):
            np.testing.assert_allclose(p, g, rtol=1e-12, atol=1e-12, err_msg=item['id'])
            error = max(error, float(np.max(np.abs(p - g), initial=0.)))
        keep = ~(item['eligible'] & (expected[0] < winner['threshold']) & (expected[1] < winner['guardian_threshold']))
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v)) for s, e, p, v in events]
        original = [vars(n).copy() for n in notes]
        midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes.copy()), SimpleNamespace(notes=[])])
        duration = sf.info(item['audio']).duration
        with patch.object(service.sf, 'read', return_value=(np.zeros(round(duration * 22050), np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=base_x):
            removed = verifier.filter('cached-dataset-evidence.wav', {}, midi)
        assert removed == int((~keep).sum()) and not midi.instruments[1].notes, item['id']
        assert [id(n) for n in midi.instruments[0].notes] == [id(n) for n, k in zip(notes, keep, strict=True) if k], item['id']
        assert original == [vars(n) for n in notes], item['id']
        count += len(notes)
        rejected += removed
    report = {'passes': True, 'recordings': len(items), 'v12_retained_events': count,
              'rejected_events': rejected, 'maximum_probability_error': error, 'tolerance': 1e-12,
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'arrays': export['sha256'], 'runtime_source_sha256': digest(Path(service.__file__)),
              'script_sha256': digest(Path(__file__)),
              'scope': 'Exact train/validation/consumed/fresh feature and sklearn probability parity; original pitch/start/end/velocity/identity/source preserved. Rejected total includes fitting and is not independent accuracy. No user uploads or scores.'}
    preserve(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    verify(args.run, args.fresh)
