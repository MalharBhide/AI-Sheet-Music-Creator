"""Broader bass training still excludes held regressions and freezes selection."""

import json

import bass_positive_release as release
import pytest
import train_bass_positive_consensus as training
from prepare_robust_training_stems import digest
from test_bass_release import frozen


def extra(tmp_path, items):
    path = tmp_path / 'extra'
    path.mkdir()
    (path / 'plan.json').write_text('{}')
    (path / 'manifest.json').write_text(json.dumps({'no_test_inference': True,
        'plan_sha256': digest(path / 'plan.json'), 'items': items}))
    return path


def test_extra_data_cannot_leak_sources_or_consume_test_clips(tmp_path, monkeypatch):
    items = [{'id': 'new', 'source_group': 'old-source', 'group': 'validation'}]
    path = extra(tmp_path, items)
    monkeypatch.setattr(training, 'load', lambda directory, group: [item for item in items if item['group'] == group])
    with pytest.raises(ValueError, match='source groups leak'):
        training.extend([{'id': 'old', 'source_group': 'old-source'}], [], [path])
    items[0]['group'] = 'test'
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['items'] = items
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Reserved/test data'):
        training.extend([], [], [path])


def test_consumed_original_stress_never_enters_fitting(tmp_path, monkeypatch):
    items = [{'id': 'bass-held-v1-repeated', 'source_group': 'stress', 'group': 'train'}]
    path = extra(tmp_path, items)
    monkeypatch.setattr(training, 'load', lambda directory, group: items if group == 'train' else [])
    with pytest.raises(ValueError, match='consumed-regression'):
        training.extend([], [], [path])


def test_expanded_frozen_loader_checks_new_policy_and_extra_manifest_hash(tmp_path):
    data, run, source, winner = frozen(tmp_path)
    path = extra(tmp_path, [])
    plan = json.loads((run / 'plan.json').read_text())
    plan.update(policy=training.POLICY, code_sha256=training.contracts(),
                additional_manifests={str(path): digest(path / 'manifest.json')})
    (run / 'plan.json').write_text(json.dumps(plan))
    winner.update(policy=training.POLICY, plan_sha256=digest(run / 'plan.json'))
    (source / 'selection.json').write_text(json.dumps(winner))
    (run / 'batch-selection.json').write_text(json.dumps({'selected': True, 'candidates': [winner],
        'winner': winner, 'plan_sha256': digest(run / 'plan.json')}))
    assert release.load_frozen(data, run)[0] == winner
    (path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='additional frozen bass data'):
        release.load_frozen(data, run)
