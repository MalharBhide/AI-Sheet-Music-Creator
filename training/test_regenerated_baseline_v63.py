from pathlib import Path

import numpy as np
import pytest

import regenerated_baseline_v63 as data


def row(identifier='example', group='train', source_group='source-a'):
    return {'id': identifier, 'group': group, 'source_group': source_group,
            'corpus': 'original-bass-timbres', 'audio': '/training/example/decoded.wav',
            'audio_sha256': 'a' * 64, 'duration': 10., 'seconds': 9.5,
            'reference': [[3., 4., 40.]], 'pitch_reference': [[3., 4., 40.]]}


def arrays():
    events = np.asarray([[3., 4., 40., 80.]])
    base = np.zeros((1, 52), np.float32)
    return {'events': events, 'base_x': base, 'x': data.context_features(events, base),
            'frames': np.zeros((1, 40, 9), np.float32),
            'attack_frames': np.zeros((1, 61, 18), np.float32),
            'release_frames': np.zeros((1, 61, 18), np.float32),
            'release_neighbors': data.neighbor_features(events, base),
            'periodicity': np.zeros((1, len(__import__('periodicity_features').NAMES)), np.float32),
            'eligible': data.eligible(events, 10.)}


def test_registry_requires_complete_disjoint_original_groups():
    rows = [row(), row('second', 'validation', 'source-b')]
    assert data.validate_registry(rows, {'train': 1, 'validation': 1}) == rows
    with pytest.raises(ValueError, match='Incomplete'):
        data.validate_registry(rows[:1], {'train': 1, 'validation': 1})
    rows[1]['source_group'] = 'source-a'
    with pytest.raises(ValueError, match='leakage'):
        data.validate_registry(rows, {'train': 1, 'validation': 1})


@pytest.mark.parametrize('field,value', [
    ('id', '../outside'), ('audio', '/outside/file.wav'),
    ('audio', '/training/../outside.wav'), ('duration', float('nan')),
    ('seconds', 11.), ('reference', [[3., 12., 40.]]),
    ('reference', [[4., 3., 40.]]), ('pitch_reference', [[3., 4., 40.5]])])
def test_source_paths_clocks_and_labels_fail_closed(field, value):
    rows = [row(), row('second', 'validation', 'source-b')]
    rows[0][field] = value
    with pytest.raises(ValueError):
        data.validate_registry(rows, {'train': 1, 'validation': 1})


def test_reserved_source_cannot_enter_fitting():
    rows = [row(), row('second', 'test', 'source-b')]
    with pytest.raises(ValueError, match='Incomplete'):
        data.validate_registry(rows, {'train': 1, 'validation': 1})


def test_pinned_witness_cannot_be_replaced(tmp_path):
    path = tmp_path / 'training/results/bass-v25-baseline-v1/manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"items": []}')
    with pytest.raises(ValueError, match='witness'):
        data.registry(tmp_path, tmp_path)


def test_same_model_architecture_normalizer_profiles_and_release_gates():
    import train_approved_finetune_v62 as old
    import train_approved_finetune_v63 as new
    assert new.PROFILES is old.PROFILES
    assert new.CHECKPOINT_SHA256 == old.CHECKPOINT_SHA256
    assert new.THRESHOLDS == old.THRESHOLDS and new.VETO_THRESHOLDS == old.VETO_THRESHOLDS
    assert new.STAGES == old.STAGES and new.FRESH_SEEDS == old.FRESH_SEEDS
    assert new.gate is old.gate and new.require_validation_proof is old.require_validation_proof
    assert new.baseline is old.baseline and new.released_models is old.released_models
    assert new.VERSION != old.VERSION
    assert new.FOLDERS['baseline'] != old.FOLDERS['baseline']


def test_musicnet_component_keeps_every_native_guard():
    # The only structural changes to the original reader are removing its parent
    # dependency and starting an empty result. Its complete component checks stay.
    source = Path(__file__).with_name('musicnet_v32_data.py').read_text()
    expected = source.replace(
        '"""Join verified real-piano observations with licensed fitting caches; forbid reserve."""',
        '"""Read the unchanged MusicNet component checks without a legacy cache dependency."""')
    expected = expected.replace('from bass_v32_data import load as previous_items\n', '')
    expected = expected.replace('def load(baseline, piano, groups=("train", "validation")):',
                                'def load(piano, groups=("train", "validation")):')
    expected = expected.replace('result = previous_items(baseline, groups)', 'result = []')
    assert Path(__file__).with_name('musicnet_component_v63.py').read_text() == expected


def test_labels_change_supervision_without_changing_model_inputs():
    values = arrays()
    positive = data.check_arrays(row(), values)
    negative_record = row()
    negative_record['reference'] = []
    negative_record['pitch_reference'] = []
    negative = data.check_arrays(negative_record, values)
    assert not np.array_equal(positive['y'], negative['y'])
    for key in data.KEYS:
        np.testing.assert_array_equal(positive[key], negative[key])


@pytest.mark.parametrize('fault', ['nonfinite', 'oversized_frame', 'float_mask', 'wrong_interval'])
def test_invalid_physical_observations_cannot_enter_fitting(fault):
    values = arrays()
    if fault == 'nonfinite': values['frames'][0, 0, 0] = np.nan
    if fault == 'oversized_frame': values['frames'][0, 0, 0] = 2.
    if fault == 'float_mask': values['eligible'] = values['eligible'].astype(float)
    if fault == 'wrong_interval': values['events'][0, 1] = 11.
    with pytest.raises(ValueError):
        data.check_arrays(row(), values)


def test_original_guitar_pitch_measurements_are_preserved_without_rounding():
    rows = [row(), row('second', 'validation', 'source-b')]
    rows[0]['corpus'] = 'guitarset-bass-register'
    rows[0]['reference'] = [[3., 4., 40.031]]
    rows[0]['pitch_reference'] = [[3., 4., 40.031]]
    assert data.validate_registry(rows, {'train': 1, 'validation': 1})[0]['reference'][0][2] == 40.031
