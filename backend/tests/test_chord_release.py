import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import chord_release as service
from app.services.chord_release_network import VERSION, model


def save(path, **changes):
    network = model(16)
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(4),
              "width": np.asarray(16), "threshold": np.asarray(.8),
              "mean": np.zeros(100, np.float32), "scale": np.ones(100, np.float32)}
    values.update({"weight_"+k: v.numpy() for k, v in network.state_dict().items()})
    values.update(changes)
    np.savez_compressed(path, **values)


@pytest.mark.parametrize("changes", [
    {"width": np.asarray(12)}, {"threshold": np.asarray(np.nan)},
    {"scale": np.zeros(100, np.float32)}, {"mean": np.zeros(84, np.float32)},
    {"unexpected": np.zeros(1)}, {"weight_ending.3.bias": np.zeros(7, np.float64)},
])
def test_portable_contract_rejects_changed_or_nonfinite_arrays(tmp_path, changes):
    path = tmp_path / "bad.npz"
    save(path, **changes)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        service.ReleaseModel(saved)


def test_asset_hash_is_checked_before_loading(tmp_path):
    path = tmp_path / "model.npz"
    save(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(PipelineError, match="damaged"):
        service.ChordRelease(path, checksum)


def test_independent_chord_endings_and_repeat_rests_are_preserved():
    events = np.array([[3, 4, 48, 80], [3, 6, 52, 90], [4.04, 5, 48, 70]], float)
    p = np.eye(7)[[6, 0, 3]]
    result = service.correct(events, p, 20, .8)
    assert result[0, 1] == 4  # An extension cannot cross a repeated-key rest.
    assert result[1, 1] == 5.92
    np.testing.assert_array_equal(result[:, [0, 2, 3]], events[:, [0, 2, 3]])


def test_a_touching_repeat_can_extend_its_own_release_without_moving_prior_attack():
    events = np.array([[3, 4, 48, 80], [4, 5, 48, 80]], float)
    result = service.correct(events, np.eye(7)[[3, 6]], 20, .8)
    np.testing.assert_array_equal(result[0], events[0])
    assert result[1, 1] == 5.08


def test_edges_and_melody_register_remain_unchanged():
    events = np.array([[1, 2, 48, 80], [18, 19, 48, 80], [3, 6, 72, 80]], float)
    np.testing.assert_array_equal(service.correct(events, np.eye(7)[[0, 0, 6]], 20, .8), events)


def test_filter_preserves_note_objects_parts_pitches_velocities_and_attacks(monkeypatch):
    verifier = service.ChordRelease.__new__(service.ChordRelease)
    verifier.model = SimpleNamespace(threshold=.8, probability=lambda item: np.eye(7)[[0, 6]])
    notes = [SimpleNamespace(start=3., end=5., pitch=48, velocity=80),
             SimpleNamespace(start=3., end=6., pitch=52, velocity=90)]
    parts = [SimpleNamespace(notes=[note]) for note in notes]
    midi = SimpleNamespace(instruments=parts)
    monkeypatch.setattr(service.sf, "read", lambda *a, **k: (np.zeros(20*22050, np.float32), 22050))
    monkeypatch.setattr(service, "note_features", lambda *a, **k: np.zeros((2, 52), np.float32))
    monkeypatch.setattr(service, "features", lambda *a: np.zeros((2, 84), np.float32))
    monkeypatch.setattr(service, "neighbors", lambda *a: np.zeros((2, 8), np.float32))
    monkeypatch.setattr(service, "observe", lambda *a: {})
    monkeypatch.setattr(service, "sequences", lambda *a: np.zeros((2, 61, 18), np.float32))
    monkeypatch.setattr(service, "release_sequences", lambda *a: np.zeros((2, 61, 18), np.float32))
    assert verifier.filter("fixture.wav", {}, midi) == 2
    assert parts[0].notes[0] is notes[0] and parts[1].notes[0] is notes[1]
    assert vars(notes[0]) == {"start": 3., "end": 4.92, "pitch": 48, "velocity": 80}
    assert vars(notes[1]) == {"start": 3., "end": 6.08, "pitch": 52, "velocity": 90}


def test_invalid_probabilities_do_not_modify_input():
    events = np.array([[3., 4., 48, 80]])
    with pytest.raises(ValueError, match="input"):
        service.correct(events, np.full((1, 7), np.nan), 20, .8)
    np.testing.assert_array_equal(events, [[3., 4., 48, 80]])
