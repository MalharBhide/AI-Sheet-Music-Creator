"""Broader offline training must not relax matched attack/hold release gates."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import train_broad_consensus as training
from test_left_consensus_v8 import fixture
from train_left_consensus_v8 import score


def test_broader_confidence_policy_still_preserves_correct_holds_and_prior_rejections():
    item = fixture()
    item['keep'][0] = False
    item['baseline_keep'] = item['keep'].copy()
    item['p'][:] = .9
    reject_false = np.array([0., 1., 0.])
    previous = score([item], [reject_false], .1)
    assert previous['passes'] and previous['false_notes_removed'] == 0
    broader = score([item], [reject_false], .1, ceiling=1.)
    assert broader['passes'] and broader['false_notes_removed'] == 1
    assert broader['per_recording'][0]['held_references_preserved']
    reject_hold = score([item], [np.zeros(3)], .1, ceiling=1.)
    assert not reject_hold['passes']
    item['shared'][2] = False
    assert score([item], [reject_false], .1, ceiling=1.)['false_notes_removed'] == 0


def test_coverage_gate_rejects_loss_of_a_continuation_even_when_attack_gate_passes():
    item = fixture()
    item['events'] = np.array([[0., 1., 48., 80.], [2., 4., 48., 80.]])
    for field in ('keep', 'baseline_keep', 'shared'):
        item[field] = np.ones(2, dtype=bool)
    item['p'] = np.full(2, .2)
    probability = [np.array([1., 0.])]
    assert score([item], probability, .1, ceiling=1.)['passes']
    guarded = score([item], probability, .1, ceiling=1., preserve_coverage=True)
    assert not guarded['passes']
    assert guarded['per_recording'][0]['lost_reference_pitch_seconds'] == 2


def test_training_excludes_ambiguous_unshared_and_already_rejected_events():
    item = fixture()
    item['y'], item['mask'] = np.array([1., 0., 0.]), np.array([True, True, False])
    item['shared'][1] = False
    x, y, weight, counts = training.matrix([item], 10.)
    assert x.shape == (1, 72) and y.tolist() == [1.] and weight.tolist() == [1.]
    assert counts[0]['positive'] == 1 and counts[0]['negative'] == 0
    item['keep'][0] = False
    with pytest.raises(ValueError, match='No supervised'):
        training.matrix([item], 10.)


def test_batch_selection_uses_only_gated_validation_gain_and_fixed_tie_order():
    rows = [{'name': 'failed', 'selected': False, 'validation_false_notes_removed': 100},
            {'name': 'zero', 'selected': True, 'validation_false_notes_removed': 0},
            {'name': 'first', 'selected': True, 'validation_false_notes_removed': 5},
            {'name': 'second', 'selected': True, 'validation_false_notes_removed': 5}]
    assert training.choose(rows)['name'] == 'first'
    assert training.choose(rows[:2]) is None


def test_external_test_labels_cannot_enter_fitting(monkeypatch, tmp_path):
    monkeypatch.setattr(training, 'load_data', lambda *_: [])
    monkeypatch.setattr(training, 'cached_external', lambda *_: [{'id': 'reserved', 'group': 'test'}])
    with pytest.raises(ValueError, match='Test data'):
        training.collect(tmp_path, [tmp_path / 'test-manifest.json'])


def test_validation_coverage_separates_confidence_from_unshared_protections():
    item = fixture()
    item['keep'][0] = False
    item['p'][1], item['p'][2] = .9, .2
    item['shared'][2] = False
    counts = training.coverage([item])
    assert counts['confident_shared_interior'] == {'matched': 1, 'unmatched': 0}
    assert counts['uncertain_shared_interior'] == {'matched': 0, 'unmatched': 0}
    assert counts['unshared_or_edge'] == {'matched': 0, 'unmatched': 1}


def frozen_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'hashes', lambda: {'website': 'fixed'})
    source = tmp_path / 'regularized'
    source.mkdir()
    plan = {'policy': training.POLICY, 'ceiling': 1., 'baseline_hashes': training.hashes(),
            'profiles': list(training.PROFILES), 'selection_safety_factor': .5,
            'pitch_range': [36, 96], 'edge_seconds': 2.5,
            'threshold_grid': training.THRESHOLDS, 'guardian_threshold_grid': training.GUARDIAN_THRESHOLDS,
            'extra_manifest_sha256': {}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    checkpoint = source / 'candidate.pickle'
    checkpoint.write_bytes(pickle.dumps([SimpleNamespace(n_features_in_=74), SimpleNamespace(n_features_in_=54)]))
    winner = {'name': 'regularized', 'selected': True, 'validation_false_notes_removed': 1,
              'plan_sha256': training.digest(tmp_path / 'plan.json'), 'threshold': .025, 'guardian_threshold': .2,
              'baseline_hashes': training.hashes(), 'policy': training.POLICY, 'ceiling': 1.,
              'checkpoint_sha256': training.digest(checkpoint)}
    (source / 'selection.json').write_text(json.dumps(winner))
    batch = {'selected': True, 'winner': winner, 'candidates': [winner], 'plan_sha256': winner['plan_sha256']}
    (tmp_path / 'batch-selection.json').write_text(json.dumps(batch))
    return winner, source


def test_frozen_batch_rejects_changed_model_selection_and_training_plan(tmp_path, monkeypatch):
    winner, source = frozen_batch(tmp_path, monkeypatch)
    assert training.load_winner(tmp_path)[0] == winner
    selection = source / 'selection.json'
    selection.write_text(json.dumps({**winner, 'threshold': .1}))
    with pytest.raises(ValueError, match='Changed frozen candidate'):
        training.load_winner(tmp_path)
    selection.write_text(json.dumps(winner))
    checkpoint = source / 'candidate.pickle'
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed after freeze'):
        training.load_winner(tmp_path)
    (tmp_path / 'plan.json').write_text('{}')
    with pytest.raises(ValueError, match='changed frozen batch'):
        training.load_winner(tmp_path)


def test_frozen_batch_checks_covered_hold_supervision_code(tmp_path, monkeypatch):
    winner, source = frozen_batch(tmp_path, monkeypatch)
    plan_path = tmp_path / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(label_policy=training.HOLD_LABELS, label_code_sha256='changed-code')
    plan_path.write_text(json.dumps(plan))
    winner['plan_sha256'] = training.digest(plan_path)
    (source / 'selection.json').write_text(json.dumps(winner))
    (tmp_path / 'batch-selection.json').write_text(json.dumps({
        'selected': True, 'winner': winner, 'candidates': [winner], 'plan_sha256': winner['plan_sha256']}))
    with pytest.raises(ValueError, match='Changed frozen held-repeat supervision'):
        training.load_winner(tmp_path)


def test_fresh_evaluation_requires_regression_gain_and_preserves_consumed_report(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'load_winner', lambda _: ({}, []))
    (tmp_path / 'regression.json').write_text(json.dumps({'passes': True, 'false_notes_removed': 0}))
    with pytest.raises(ValueError, match='positive regression'):
        training.test(tmp_path, tmp_path, tmp_path / 'fresh-manifest.json')
    (tmp_path / 'fresh.json').write_text('{}')
    with pytest.raises(ValueError, match='Preserve consumed'):
        training.test(tmp_path, tmp_path, tmp_path / 'fresh-manifest.json')


def test_failed_or_missing_coverage_blocks_fresh_test_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'load_winner', lambda _: ({'checkpoint_sha256': 'frozen'}, []))
    (tmp_path / 'regression.json').write_text(json.dumps({'passes': True, 'false_notes_removed': 1}))
    with pytest.raises(ValueError, match='until reference coverage passes'):
        training.test(tmp_path, tmp_path, tmp_path / 'fresh-manifest.json')
    (tmp_path / 'coverage-release-gate.json').write_text(json.dumps({'passes': False}))
    with pytest.raises(ValueError, match='Failed or changed reference coverage'):
        training.test(tmp_path, tmp_path, tmp_path / 'fresh-manifest.json')
    (tmp_path / 'coverage-release-gate.json').write_text(json.dumps({
        'passes': True, 'checkpoint_sha256': 'changed'}))
    with pytest.raises(ValueError, match='Failed or changed reference coverage'):
        training.test(tmp_path, tmp_path, tmp_path / 'fresh-manifest.json')
