"""Residual release pins the shipped baseline and its distinct feature contract."""

import json
import pickle
from types import SimpleNamespace

import bass_residual_v11_release as release
import pytest
from prepare_robust_training_stems import digest
from test_bass_boundary_release import frozen


@pytest.fixture(autouse=True)
def isolated_historical_route_contract(monkeypatch):
    # These unit fixtures exercise V11 logic with mocked data. Permit today's
    # route only within the fixture; real experiment contracts remain frozen.
    from pathlib import Path

    import current_bass_v11_baseline as historical
    from prepare_robust_training_stems import digest

    route = 'backend/app/services/piano_transcription.py'
    monkeypatch.setitem(historical.SOURCES, route, digest(Path(__file__).resolve().parents[1] / route))


def residual(tmp_path):
    run, old, data, winner = frozen(tmp_path)
    source = old.with_name('precision')
    old.rename(source)
    plan = json.loads((run / 'plan.json').read_text())
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text('{}')
    (cache / 'plan.json').write_text('{}')
    plan.update(version=release.VERSION, baseline_hashes=release.hashes(),
                guardian_positive_weight=release.GUARDIAN_POSITIVE_WEIGHT,
                baseline_cache_root=str(cache),
                baseline_cache_manifest_sha256=digest(cache / 'manifest.json'),
                baseline_cache_plan_sha256=digest(cache / 'plan.json'), code_sha256=release.contracts(), profiles=list(release.PROFILES),
                stages=list(release.STAGES), threshold_grid=list(release.THRESHOLDS), guardian_grid=list(release.GUARDIANS),
                feature_counts=[74, 28], feature_names=list(release.NAMES), guardian_names=list(release.ACOUSTIC_NAMES))
    (run / 'plan.json').write_text(json.dumps(plan))
    with (source / 'candidate.pickle').open('wb') as stream:
        pickle.dump([SimpleNamespace(n_features_in_=count, _predictors=[None] * 100) for count in (74, 28)], stream)
    winner.update(name='precision', plan_sha256=digest(run / 'plan.json'),
                  checkpoint_sha256=digest(source / 'candidate.pickle'))
    (source / 'selection.json').write_text(json.dumps(winner))
    batch = {'selected': True, 'winner': winner, 'candidates': [winner],
             'plan_sha256': winner['plan_sha256'], 'test_used_for_selection': False}
    (run / 'batch-selection.json').write_text(json.dumps(batch))
    return run, source, data, winner


def test_historical_boundary_weights_cannot_be_mistaken_for_residual_model(tmp_path):
    run, _, _, _ = frozen(tmp_path)
    with pytest.raises(ValueError, match='plan or winner'):
        release.load_frozen(run)


def test_residual_checkpoint_binds_distinct_feature_and_dataset_contract(tmp_path):
    run, _, data, winner = residual(tmp_path)
    assert release.load_frozen(run)[1] == winner
    (data / 'manifest.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='frozen bass boundary dataset'):
        release.load_frozen(run)


def test_failed_regression_cannot_be_replaced_by_another_threshold(tmp_path):
    run, _, _, _ = residual(tmp_path)
    (run / 'consumed-regression.json').write_text('{}')
    with pytest.raises(ValueError, match='no runner-up or test tuning'):
        release.evaluate(run)


@pytest.mark.parametrize('artifact', ['manifest', 'plan'])
def test_v11_duration_cache_cannot_change_after_selection(tmp_path, artifact):
    run, _, _, _ = residual(tmp_path)
    assert release.load_frozen(run)[0]['feature_counts'] == [74, 28]
    (tmp_path / 'cache' / (artifact + '.json')).write_text('{"changed": true}')
    with pytest.raises(ValueError, match='V11 baseline cache'):
        release.load_frozen(run)
