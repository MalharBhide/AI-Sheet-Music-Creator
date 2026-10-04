"""Bass decoder scope and per-recording attack/hold gates cannot be relaxed."""

import hashlib
import json
from types import SimpleNamespace

import bass_training_data as data
import numpy as np
import pytest
import soundfile as sf
import train_bass_consensus as training
from prepare_slakh_bass import bass_sources


def fixture():
    events = np.array([[3., 5., 40., 90.], [5., 7., 40., 85.], [8., 9., 52., 70.]])
    return {'id': 'bass', 'corpus': 'fixture', 'seconds': 20., 'events': events,
            'reference': np.array([[3., 5., 40.], [5., 7., 40.]]),
            'pitch_reference': np.array([[3., 5., 40.], [5., 7., 40.]]),
            'eligible': np.ones(3, dtype=bool)}


def test_consensus_rejects_wrong_note_without_changing_real_repeated_notes():
    item = fixture()
    before = item['events'].copy()
    confidence = np.array([1., 1., 0.])
    chosen, _ = training.select([item], [confidence], [confidence], {})
    assert chosen['passes'] and chosen['false_notes_removed'] == 1
    np.testing.assert_array_equal(item['events'], before)
    row = chosen['per_recording'][0]
    assert row['matched_references_preserved'] and row['held_references_preserved']
    # A single head cannot erase a real or false candidate by itself.
    chosen, _ = training.select([item], [confidence], [np.ones(3)], {})
    assert chosen is None


def test_real_repeated_attack_loss_blocks_selection_even_with_more_false_notes_removed():
    item = fixture()
    p = np.array([1., 0., 0.])
    chosen, _ = training.select([item], [p], [p], {})
    assert chosen is None


def test_unmatched_hold_fragment_and_crossing_crop_are_protected_by_pitch_coverage():
    item = fixture()
    item['reference'] = item['reference'][:1]
    item['pitch_reference'] = np.array([[0., 7., 40.]])
    p = np.array([1., 0., 0.])
    result = training.score([item], [p], [p], .1, .1)
    row = result['per_recording'][0]
    assert row['matched_references_preserved'] and row['held_references_preserved']
    assert not row['reference_pitch_coverage_preserved'] and not result['passes']
    assert row['lost_reference_pitch_seconds'] == 2.


def test_edges_and_frequency_bounds_keep_incomplete_or_nonbass_candidates():
    events = np.array([[1., 3., 40., 90.], [3., 28., 40., 90.], [3., 4., 60., 90.],
                       [3., 4., 21., 90.], [3., 4., 59., 90.]])
    np.testing.assert_array_equal(data.eligible(events, 30), [False, False, False, True, True])
    item = fixture()
    item['eligible'][:] = False
    result = training.score([item], [np.zeros(3)], [np.zeros(3)], .1, .1)
    assert result['passes'] and result['false_notes_removed'] == 0
    with pytest.raises(ValueError, match='paired bass confidence'):
        training.score([item], [np.full(3, np.nan)], [np.zeros(3)], .1, .1)


def test_generic_bass_cache_uses_production_constraints_and_refuses_changed_identity(tmp_path, monkeypatch):
    audio = tmp_path / 'bass.wav'
    sf.write(audio, np.zeros(22050 * 10), 22050, subtype='FLOAT')
    item = {'id': 'unit', 'group': 'train', 'source_group': 'unit', 'corpus': 'fixture',
            'audio': str(audio), 'reference': [[3., 4., 40.]],
            'pitch_reference': [[3., 4., 40.]], 'evaluation_window': [.25, 9.75]}
    calls = []
    monkeypatch.setattr(data, '_MODEL', object())

    def predict(path, **kwargs):
        calls.append(kwargs)
        return {}, SimpleNamespace(instruments=[SimpleNamespace(notes=[
            SimpleNamespace(start=3., end=4., pitch=40, velocity=90)])]), []

    monkeypatch.setattr(data, 'predict', predict)
    monkeypatch.setattr(data, 'note_features', lambda *args, **kwargs: np.ones((1, 52), dtype=np.float32))
    path = data.cache_one(tmp_path, item)
    assert data.cache_one(tmp_path, item) == path and len(calls) == 1
    config = calls[0]
    assert config['minimum_frequency'] == pytest.approx(27.5)
    assert config['maximum_frequency'] == pytest.approx(261.6255653)
    assert config['onset_threshold'] == .5 and config['frame_threshold'] == .3
    assert config['minimum_note_length'] == 90 and not config['melodia_trick']
    manifest = {'items': [item], 'cache_sha256': {'unit': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert data.load(tmp_path, 'validation') == []
    assert len(data.load(tmp_path, 'train')) == 1
    with pytest.raises(ValueError, match='cache identity'):
        data.cache_one(tmp_path, {**item, 'evaluation_window': [1., 9.]})
    path.write_bytes(b'damaged')
    with pytest.raises(ValueError, match='frozen bass features'):
        data.load(tmp_path, 'train')


def test_midi_program_selection_excludes_drums_and_nonbass_instruments():
    metadata = {'stems': {'piano': {'is_drum': False, 'program_num': 0},
                          'bass': {'is_drum': False, 'program_num': 32},
                          'synthbass': {'is_drum': False, 'program_num': 39},
                          'violin': {'is_drum': False, 'program_num': 40},
                          'drum': {'is_drum': True, 'program_num': 32}}}
    assert bass_sources(metadata) == ['bass', 'synthbass']
