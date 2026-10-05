"""Dataset continuation cannot fit partial data or silently switch baselines."""

import json
from pathlib import Path

import continue_bass_v11_training as runner
import pytest
from prepare_robust_training_stems import digest


@pytest.fixture(autouse=True)
def isolated_historical_route_contract(monkeypatch):
    # These unit fixtures exercise V11 logic with mocked data. Permit today's
    # route only within the fixture; real experiment contracts remain frozen.
    from pathlib import Path

    import current_bass_v11_baseline as historical
    from prepare_robust_training_stems import digest

    route = 'backend/app/services/piano_transcription.py'
    monkeypatch.setitem(historical.SOURCES, route, digest(Path(__file__).resolve().parents[1] / route))


def complete(tmp_path):
    root = tmp_path / 'slakh-bass-demucs-expanded-v1'
    root.mkdir()
    plan = {'demucs_offsets': [30, 60, 90], 'oracle_offsets': [],
            'code_sha256': {'continue_bass_v11_training.py': digest(Path(runner.__file__))}}
    (root / 'plan.json').write_text(json.dumps(plan))
    manifest = {'plan_sha256': digest(root / 'plan.json'), 'no_test_inference': True,
                'no_user_audio_or_scores': True,
                'items': [{'group': group} for group, count in [('train', 36), ('validation', 12)] for _ in range(count)]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def test_only_complete_fixed_partition_can_start_baseline_preparation(tmp_path):
    assert not runner.ready(tmp_path)
    root = complete(tmp_path)
    assert runner.ready(root)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['items'].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        runner.ready(root)


def test_stopped_producer_without_manifest_records_failure_and_never_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'prepare_original', lambda _: None)
    monkeypatch.setattr(runner, 'producer_running', lambda: False)
    monkeypatch.setattr(runner, 'train', lambda *_: pytest.fail('Partial data entered fitting'))
    with pytest.raises(RuntimeError, match='without its complete manifest'):
        runner.run(tmp_path)
    status = json.loads((tmp_path / 'bass-expanded-training-v3/status.json').read_text())
    assert status['state'] == 'failed' and not status['production_weights_changed']


def test_changed_fitting_contract_stops_before_caching_or_fitting(tmp_path, monkeypatch):
    complete(tmp_path)
    monkeypatch.setattr(runner, 'prepare_original', lambda _: None)
    calls = iter([{'version': 'one'}, {'version': 'two'}])
    monkeypatch.setattr(runner, 'contracts', lambda: next(calls))
    monkeypatch.setattr(runner, 'prepare_baseline', lambda *_: pytest.fail('Changed contract entered caching'))
    with pytest.raises(ValueError, match='training contract'):
        runner.run(tmp_path)
    status = json.loads((tmp_path / 'bass-expanded-training-v3/status.json').read_text())
    assert status['state'] == 'failed' and status['user_uploads_processed'] == 0
