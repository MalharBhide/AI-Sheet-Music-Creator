"""Matched attacks alone are insufficient evidence to release a new classifier."""

import json
from pathlib import Path

import pytest
from export_broad_consensus import release_reports
from train_broad_consensus import digest


def reports(tmp_path):
    winner = {'name': 'shallow', 'checkpoint_sha256': 'frozen', 'threshold': .005, 'guardian_threshold': .05}
    (tmp_path / 'shallow').mkdir()
    (tmp_path / 'shallow/selection.json').write_text(json.dumps(winner))
    (tmp_path / 'batch-selection.json').write_text(json.dumps({'winner': winner}))
    (tmp_path / 'plan.json').write_text(json.dumps({'preserve_reference_pitch_coverage': True}))
    row = {'passes': True, 'reference_pitch_coverage_preserved': True}
    report = {'passes': True, 'checkpoint_sha256': 'frozen', 'threshold': .005, 'guardian_threshold': .05,
              'batch_selection_sha256': digest(tmp_path / 'batch-selection.json'),
              'selection_sha256': digest(tmp_path / 'shallow/selection.json'),
              'per_recording': [row], 'false_notes_removed': 1}
    for stage in ('regression', 'fresh'):
        (tmp_path / (stage + '.json')).write_text(json.dumps(report))
    coverage = {'passes': True, 'checkpoint_sha256': 'frozen',
                'batch_selection_sha256': report['batch_selection_sha256'], 'per_recording': [{'passes': True}],
                'gate_code_sha256': digest(Path(__file__).with_name('check_reference_coverage.py')),
                'coverage_code_sha256': digest(Path(__file__).with_name('pitch_interval_coverage.py'))}
    (tmp_path / 'coverage-release-gate.json').write_text(json.dumps(coverage))
    return winner, report


def test_export_requires_all_three_frozen_preservation_stages(tmp_path):
    winner, report = reports(tmp_path)
    release_reports(tmp_path, winner)
    report['per_recording'][0]['reference_pitch_coverage_preserved'] = False
    (tmp_path / 'fresh.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed frozen'):
        release_reports(tmp_path, winner)


def test_export_rejects_zero_gain_changed_thresholds_and_changed_coverage(tmp_path):
    winner, report = reports(tmp_path)
    report['false_notes_removed'] = 0
    (tmp_path / 'regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='improve'):
        release_reports(tmp_path, winner)
    report['false_notes_removed'], report['threshold'] = 1, .1
    (tmp_path / 'regression.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Failed or changed frozen'):
        release_reports(tmp_path, winner)
