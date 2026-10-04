"""A failed, changed, or unselected bass model cannot consume reserved tests."""

import json
import pickle
from types import SimpleNamespace

import export_bass_consensus
import numpy as np
import prepare_bass_regression
import prepare_slakh_bass_test
import pytest
from bass_release import load_frozen, require_report
from export_context_verifier import export
from prepare_bass_regression import CASES, RATE, SECONDS, render, score
from prepare_robust_training_stems import digest
from sklearn.ensemble import HistGradientBoostingClassifier
from train_bass_consensus import (
    GUARDIANS,
    POLICY,
    PROFILES,
    STAGES,
    THRESHOLDS,
    at_stage,
    contracts,
)


def frozen(tmp_path):
    data, run = tmp_path / 'data', tmp_path / 'run'
    data.mkdir()
    run.mkdir()
    (data / 'manifest.json').write_text('{}')
    (data / 'plan.json').write_text('{}')
    plan = {'policy': POLICY, 'profiles': list(PROFILES), 'stages': list(STAGES),
            'threshold_grid': list(THRESHOLDS), 'guardian_grid': list(GUARDIANS), 'margin': .5,
            'code_sha256': contracts(), 'manifest_sha256': digest(data / 'manifest.json'),
            'preparation_sha256': digest(data / 'plan.json')}
    (run / 'plan.json').write_text(json.dumps(plan))
    source = run / 'regularized'
    source.mkdir()
    models = [SimpleNamespace(n_features_in_=count, _predictors=[None] * 100) for count in (52, 26)]
    with (source / 'candidate.pickle').open('wb') as stream:
        pickle.dump(models, stream)
    winner = {'name': 'regularized', 'selected': True, 'validation_false_notes_removed': 2,
              'validation_log_loss': .3, 'trees': 100, 'policy': POLICY,
              'plan_sha256': digest(run / 'plan.json'), 'threshold': .01, 'guardian_threshold': .025,
              'checkpoint_sha256': digest(source / 'candidate.pickle')}
    (source / 'selection.json').write_text(json.dumps(winner))
    (source / 'validation.json').write_text(json.dumps({'passes': True, 'per_recording': [{'passes': True}],
        'threshold': .01, 'guardian_threshold': .025, 'false_notes_removed': 2}))
    batch = {'candidates': [winner], 'selected': True, 'winner': winner, 'plan_sha256': digest(run / 'plan.json')}
    (run / 'batch-selection.json').write_text(json.dumps(batch))
    return data, run, source, winner


def test_frozen_loader_rejects_changed_checkpoints_selection_or_contracts(tmp_path):
    data, run, source, winner = frozen(tmp_path)
    selected, models = load_frozen(data, run)
    assert selected == winner and [m.n_features_in_ for m in models] == [52, 26]
    (source / 'candidate.pickle').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='checkpoint'):
        load_frozen(data, run)


def test_zero_gain_or_failed_validation_cannot_consume_tests(tmp_path):
    data, run, source, winner = frozen(tmp_path)
    validation = json.loads((source / 'validation.json').read_text())
    validation['per_recording'][0]['passes'] = False
    (source / 'validation.json').write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='per-recording bass validation'):
        load_frozen(data, run)
    batch = json.loads((run / 'batch-selection.json').read_text())
    batch['candidates'][0]['validation_false_notes_removed'] = 0
    (run / 'batch-selection.json').write_text(json.dumps(batch))
    with pytest.raises(ValueError, match='positive frozen'):
        load_frozen(data, run)


def test_release_report_requires_every_preservation_gate(tmp_path):
    _, run, _, winner = frozen(tmp_path)
    result = {'passes': True, 'per_recording': [{'passes': True, 'matched_references_preserved': True,
              'held_references_preserved': True, 'reference_pitch_coverage_preserved': False}],
              'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'false_notes_removed': 2}
    (run / 'held-regression.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='release evidence'):
        require_report(run, winner, 'held-regression')


def test_stress_fixtures_have_exact_clocks_repeats_long_holds_and_simultaneous_octaves():
    for index, case in enumerate(CASES):
        notes = score(case)
        audio = render(notes, index)
        assert len(audio) == RATE * SECONDS and np.isfinite(audio).all()
        assert all(0 <= start < end <= SECONDS and 21 <= pitch < 60 for start, end, pitch in notes)
        assert np.count_nonzero(audio[:3 * RATE]) == 0
    assert max(end - start for start, end, _ in score('held')) == 8
    repeats = score('repeated')
    np.testing.assert_allclose(np.diff([start for start, *_ in repeats]), .5)
    np.testing.assert_allclose(np.diff([start for start, *_ in score('triplet')]), 1 / 3)
    assert score('octave')[0][:2] == score('octave')[1][:2]
    with pytest.raises(ValueError, match='predeclared'):
        score('a user song')


def test_selected_earlier_checkpoint_matches_public_staged_predictions_and_portable_arrays(tmp_path):
    # This checks sklearn's private-prefix representation against its public
    # staged interface, then independently traverses exported numeric trees.
    x = np.random.default_rng(149).normal(size=(120, 52)).astype(np.float32)
    y = (x[:, 0] + .5 * x[:, 1] > 0).astype(float)
    model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=7,
        min_samples_leaf=10, early_stopping=False, random_state=191).fit(x, y)
    earlier = at_stage([model], 100)[0]
    expected = next(prediction[:, 1] for stage, prediction in enumerate(model.staged_predict_proba(x), 1)
                    if stage == 100)
    np.testing.assert_allclose(earlier.predict_proba(x)[:, 1], expected, atol=1e-12, rtol=1e-12)
    assert len(model._predictors) == 200 and len(earlier._predictors) == 100
    export(earlier, .01, tmp_path / 'test-arrays.npz', [{'x': x}])
    with pytest.raises(ValueError, match='frozen training stages'):
        at_stage([model], 500)


def test_no_validation_winner_or_failed_stress_does_not_read_reserved_audio(tmp_path, monkeypatch):
    def blocked(*args):
        raise ValueError('No passing frozen winner')

    monkeypatch.setattr(prepare_bass_regression, 'load_frozen', blocked)
    with pytest.raises(ValueError, match='passing frozen'):
        prepare_bass_regression.prepare(tmp_path / 'missing-data', tmp_path / 'missing-run')
    monkeypatch.setattr(prepare_slakh_bass_test, 'load_frozen', lambda *args: ({}, []))
    monkeypatch.setattr(prepare_slakh_bass_test, 'require_report', blocked)
    with pytest.raises(ValueError, match='passing frozen'):
        prepare_slakh_bass_test.prepare(tmp_path / 'missing-data', tmp_path / 'missing-run')
    assert list(tmp_path.iterdir()) == []


def test_failed_reserved_gate_cannot_export_or_create_portable_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(export_bass_consensus, 'load_frozen', lambda *args: ({}, []))

    def report(*args, **kwargs):
        if kwargs.get('positive'):
            raise ValueError('No positive preservation-safe reserved gain')

    monkeypatch.setattr(export_bass_consensus, 'require_report', report)
    with pytest.raises(ValueError, match='reserved gain'):
        export_bass_consensus.run_export(tmp_path / 'missing-data', tmp_path / 'missing-run')
    assert list(tmp_path.iterdir()) == []
