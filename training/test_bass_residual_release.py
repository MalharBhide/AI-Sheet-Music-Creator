"""Residual release pins the shipped baseline and its distinct feature contract."""

import json
import pickle
from types import SimpleNamespace

import bass_residual_release as release
import pytest
from prepare_robust_training_stems import digest
from test_bass_boundary_release import frozen


def residual(tmp_path):
    run, old, data, winner = frozen(tmp_path)
    source = old.with_name('regularized')
    old.rename(source)
    plan = json.loads((run / 'plan.json').read_text())
    plan.update(version=release.VERSION, code_sha256=release.contracts(), profiles=list(release.PROFILES),
                stages=list(release.STAGES), threshold_grid=list(release.THRESHOLDS), guardian_grid=list(release.GUARDIANS),
                feature_counts=[74, 54], feature_names=list(release.NAMES), guardian_names=list(release.ACOUSTIC_NAMES))
    (run / 'plan.json').write_text(json.dumps(plan))
    with (source / 'candidate.pickle').open('wb') as stream:
        pickle.dump([SimpleNamespace(n_features_in_=count, _predictors=[None] * 100) for count in (74, 54)], stream)
    winner.update(name='regularized', plan_sha256=digest(run / 'plan.json'),
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
