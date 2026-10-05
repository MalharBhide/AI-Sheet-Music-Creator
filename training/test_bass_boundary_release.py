"""A frozen winner is the only checkpoint allowed through consumed evaluation."""

import json
import pickle
from types import SimpleNamespace

import bass_boundary_release as release
import pytest
from prepare_robust_training_stems import digest


def frozen(tmp_path):
    run = tmp_path / 'run'
    source = run / 'guarded'
    source.mkdir(parents=True)
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text('{}')
    plan = {'version': release.VERSION, 'baseline_hashes': release.hashes(), 'code_sha256': release.contracts(),
            'manifests': {str(data): digest(data / 'manifest.json')}, 'consumed_regression_manifests': {},
            'profiles': list(release.PROFILES), 'stages': list(release.STAGES),
            'threshold_grid': list(release.THRESHOLDS), 'guardian_grid': list(release.GUARDIANS),
            'safety_factor': .5, 'feature_counts': [112, 60]}
    (run / 'plan.json').write_text(json.dumps(plan))
    with (source / 'candidate.pickle').open('wb') as stream:
        pickle.dump([SimpleNamespace(n_features_in_=count, _predictors=[None] * 100) for count in (112, 60)], stream)
    validation = {'passes': True, 'per_recording': [{'passes': True}],
                  'threshold': .005, 'guardian_threshold': .05, 'false_notes_removed': 2}
    (source / 'validation.json').write_text(json.dumps(validation))
    winner = {'name': 'guarded', 'selected': True, 'validation_false_notes_removed': 2,
              'validation_log_loss': .3, 'trees': 100, 'threshold': .005, 'guardian_threshold': .05,
              'test_used_for_selection': False, 'plan_sha256': digest(run / 'plan.json'),
              'checkpoint_sha256': digest(source / 'candidate.pickle'),
              'validation_sha256': digest(source / 'validation.json')}
    (source / 'selection.json').write_text(json.dumps(winner))
    (run / 'batch-selection.json').write_text(json.dumps({'selected': True, 'winner': winner,
        'candidates': [winner], 'plan_sha256': digest(run / 'plan.json'), 'test_used_for_selection': False}))
    return run, source, data, winner


@pytest.mark.parametrize('artifact', ['manifest', 'checkpoint', 'validation', 'selection'])
def test_changed_data_or_model_cannot_pass_frozen_release(tmp_path, artifact):
    run, source, data, winner = frozen(tmp_path)
    assert release.load_frozen(run)[1] == winner
    path = {'manifest': data / 'manifest.json', 'checkpoint': source / 'candidate.pickle',
            'validation': source / 'validation.json', 'selection': source / 'selection.json'}[artifact]
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Changed frozen'):
        release.load_frozen(run)


def test_runner_up_and_zero_gain_cannot_replace_selected_winner(tmp_path):
    run, _, _, _ = frozen(tmp_path)
    path = run / 'batch-selection.json'
    batch = json.loads(path.read_text())
    batch['winner'] = {**batch['winner'], 'name': 'recall'}
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match='plan or winner'):
        release.load_frozen(run)
    batch['selected'] = False
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match='No positive'):
        release.load_frozen(run)


def test_consumed_regression_cannot_be_repeated_or_used_for_retuning(tmp_path):
    run, _, _, _ = frozen(tmp_path)
    (run / 'consumed-regression.json').write_text('{}')
    with pytest.raises(ValueError, match='no runner-up or test tuning'):
        release.evaluate(run)
