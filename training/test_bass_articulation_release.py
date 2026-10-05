"""The next label policy cannot silently reuse a historical failed winner."""

import json

import bass_articulation_release as release
import pytest
from prepare_robust_training_stems import digest
from test_bass_boundary_release import frozen


def test_historical_boundary_policy_is_not_a_release_aware_checkpoint(tmp_path):
    run, _, _, _ = frozen(tmp_path)
    with pytest.raises(ValueError, match='plan or winner'):
        release.load_frozen(run)


def test_new_policy_binds_its_own_code_and_keeps_shape_and_winner_checks(tmp_path):
    run, source, _, winner = frozen(tmp_path)
    plan = json.loads((run / 'plan.json').read_text())
    plan.update(version=release.VERSION, code_sha256=release.contracts())
    (run / 'plan.json').write_text(json.dumps(plan))
    winner['plan_sha256'] = digest(run / 'plan.json')
    (source / 'selection.json').write_text(json.dumps(winner))
    batch = {'selected': True, 'winner': winner, 'candidates': [winner],
             'plan_sha256': winner['plan_sha256'], 'test_used_for_selection': False}
    (run / 'batch-selection.json').write_text(json.dumps(batch))
    assert release.load_frozen(run)[1] == winner
    plan['code_sha256']['training/bass_articulation_labels.py'] = 'stale'
    (run / 'plan.json').write_text(json.dumps(plan))
    winner['plan_sha256'] = digest(run / 'plan.json')
    (source / 'selection.json').write_text(json.dumps(winner))
    batch['plan_sha256'] = winner['plan_sha256']
    (run / 'batch-selection.json').write_text(json.dumps(batch))
    with pytest.raises(ValueError, match='plan or winner'):
        release.load_frozen(run)
