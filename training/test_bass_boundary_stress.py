"""Original first-pass stress cannot use failed models or fitting sources."""

import json

import prepare_bass_boundary_stress as stress
import pytest
from prepare_robust_training_stems import digest


def regression(run):
    (run / 'batch-selection.json').write_text('{}')
    winner = {'checkpoint_sha256': 'frozen', 'threshold': .05, 'guardian_threshold': .05}
    report = {'passes': True, 'per_recording': [{'passes': True}],
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'threshold': .05, 'guardian_threshold': .05}
    (run / 'consumed-regression.json').write_text(json.dumps(report))
    return winner, report


def test_failed_or_different_checkpoint_stops_before_stress_inference(tmp_path):
    winner, report = regression(tmp_path)
    assert stress.require_regression(tmp_path, winner) == digest(tmp_path / 'consumed-regression.json')
    report['per_recording'][0]['passes'] = False
    (tmp_path / 'consumed-regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed'):
        stress.require_regression(tmp_path, winner)
    report['per_recording'][0]['passes'] = True
    report['checkpoint_sha256'] = 'another-model'
    (tmp_path / 'consumed-regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed'):
        stress.require_regression(tmp_path, winner)


def test_fitting_seed_cannot_be_reclassified_as_first_pass_test(tmp_path, monkeypatch):
    winner, _ = regression(tmp_path)
    data = tmp_path / 'fitted'
    data.mkdir()
    (data / 'manifest.json').write_text(json.dumps({'items': [{'source_group': 'original-bass-seed-' + str(stress.SEEDS[0])}]}))
    monkeypatch.setattr(stress, 'load_frozen', lambda run: ({'manifests': {str(data): 'unused'}}, winner, []))
    monkeypatch.setattr(stress, 'cache_one', lambda *args: pytest.fail('Overlapping test inference'))
    with pytest.raises(ValueError, match='source already fitted'):
        stress.prepare(tmp_path, tmp_path / 'stress')


def test_consumed_first_pass_cannot_be_repeated(tmp_path):
    (tmp_path / 'original-first-pass.json').write_text('{}')
    with pytest.raises(ValueError, match='no retuning'):
        stress.evaluate(tmp_path, tmp_path / 'unused')
