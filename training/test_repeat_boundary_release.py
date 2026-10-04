import json

import pytest
from repeat_boundary_release import require_report
from train_broad_consensus import digest


def evidence(tmp_path):
    (tmp_path / 'batch-selection.json').write_text('{}')
    winner = {'checkpoint_sha256': 'frozen', 'threshold': .05, 'guardian_threshold': .025}
    report = {**winner, 'batch_selection_sha256': digest(tmp_path / 'batch-selection.json'),
              'passes': True, 'false_notes_removed': 2, 'per_recording': [{
                  'passes': True, 'matched_references_preserved': True, 'held_references_preserved': True,
                  'reference_pitch_coverage_preserved': True, 'pitch_time_union_preserved': True}]}
    (tmp_path / 'regression.json').write_text(json.dumps(report))
    return winner, report


@pytest.mark.parametrize('field', ['matched_references_preserved', 'held_references_preserved',
                                  'reference_pitch_coverage_preserved', 'pitch_time_union_preserved'])
def test_boundary_release_requires_every_preservation_gate(tmp_path, field):
    winner, report = evidence(tmp_path)
    require_report(tmp_path, winner, 'regression', positive=True)
    report['per_recording'][0][field] = False
    (tmp_path / 'regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed'):
        require_report(tmp_path, winner, 'regression', positive=True)


@pytest.mark.parametrize('mutation', ['empty', 'zero-gain', 'changed-threshold', 'changed-model'])
def test_boundary_release_rejects_incomplete_or_changed_evidence(tmp_path, mutation):
    winner, report = evidence(tmp_path)
    if mutation == 'empty':
        report['per_recording'] = []
    elif mutation == 'zero-gain':
        report['false_notes_removed'] = 0
    elif mutation == 'changed-threshold':
        report['threshold'] = .1
    else:
        report['checkpoint_sha256'] = 'changed'
    (tmp_path / 'regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed'):
        require_report(tmp_path, winner, 'regression', positive=True)
