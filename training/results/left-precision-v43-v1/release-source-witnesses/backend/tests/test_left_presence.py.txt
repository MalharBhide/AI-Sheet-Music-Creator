import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import left_presence as service
from app.services.left_presence_network import VERSION, gated, keep, model


def save(path, **changes):
    network = model(12)
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(5),
        "width": np.asarray(12), "threshold": np.asarray(.9), "veto_threshold": np.asarray(.2),
        "mean": np.zeros(163, np.float32), "scale": np.ones(163, np.float32)}
    values.update({"weight_"+key: value.numpy() for key, value in network.state_dict().items()})
    values.update(changes)
    np.savez_compressed(path, **values)


@pytest.mark.parametrize("changes", [
    {"width": np.asarray(20)}, {"threshold": np.asarray(.89)}, {"veto_threshold": np.asarray(np.nan)},
    {"scale": np.zeros(163, np.float32)}, {"mean": np.zeros(100, np.float32)},
    {"unexpected": np.zeros(1)}, {"weight_frame.bias": np.zeros(1, np.float64)},
])
def test_changed_export_contract_cannot_load(tmp_path, changes):
    path = tmp_path/"bad.npz"
    save(path, **changes)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        service.PresenceModel(saved)


def test_hash_checked_before_inference(tmp_path):
    path = tmp_path/"model.npz"
    save(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(PipelineError, match="damaged"):
        service.LeftHandPresence(path, checksum)


def test_one_supported_central_frame_vetoes_deletion_even_for_short_overlap():
    rejection = np.array([[.001, .999]]*3)
    presence = np.zeros((3, 40))
    presence[0, 8], presence[1, 31], presence[2, 7] = .2, .9, 1.
    actual = gated((rejection, presence), .2)
    np.testing.assert_array_equal(actual[:2], [[1., 0.], [1., 0.]])
    np.testing.assert_array_equal(actual[2], rejection[2])


def test_removal_preserves_edges_melody_and_every_retained_clock():
    events = np.array([[1, 2, 48, 80], [18, 19, 48, 80], [3, 6, 72, 90],
                       [3, 6, 48, 80], [6.04, 7, 48, 70]], float)
    accepted = keep(events, np.array([[.01, .99]]*5), 20, .9)
    np.testing.assert_array_equal(accepted, [True, True, True, False, False])
    np.testing.assert_array_equal(events[accepted], events[:3])


def test_prepared_deletion_never_changes_release_context_or_retained_chords(monkeypatch):
    verifier = service.LeftHandPresence.__new__(service.LeftHandPresence)
    notes = [SimpleNamespace(start=3., end=4., pitch=48, velocity=80),
             SimpleNamespace(start=3., end=6., pitch=52, velocity=90),
             SimpleNamespace(start=4.04, end=5., pitch=48, velocity=70)]
    parts = [SimpleNamespace(notes=notes[:2]), SimpleNamespace(notes=notes[2:])]
    midi = SimpleNamespace(instruments=parts)
    captured = []

    def decide(item, duration):
        captured.append(item["events"].copy())
        assert duration == 20
        return np.array([True, False, True])

    verifier.model = SimpleNamespace(keep=decide)
    monkeypatch.setattr(service.sf, "read", lambda *a, **k: (np.zeros(20*22050, np.float32), 22050))
    monkeypatch.setattr(service, "note_features", lambda *a, **k: np.zeros((3, 52), np.float32))
    monkeypatch.setattr(service, "features", lambda *a: np.zeros((3, 84), np.float32))
    monkeypatch.setattr(service, "neighbors", lambda *a: np.zeros((3, 8), np.float32))
    monkeypatch.setattr(service, "observe", lambda *a: {"cqt": None})
    monkeypatch.setattr(service, "coarse_sequences", lambda *a: np.zeros((3, 40, 9), np.float32))
    monkeypatch.setattr(service, "sequences", lambda *a: np.zeros((3, 61, 18), np.float32))
    monkeypatch.setattr(service, "release_sequences", lambda *a: np.zeros((3, 61, 18), np.float32))
    monkeypatch.setattr(service, "recurrence", lambda *a: np.zeros((3, 63), np.float32))
    proposal = verifier.prepare("fixture.wav", {}, midi)
    assert parts[0].notes == notes[:2] and parts[1].notes == notes[2:]
    np.testing.assert_array_equal(captured[0], [[3., 4., 48, 80], [3., 6., 52, 90], [4.04, 5., 48, 70]])
    # The independent release stage still receives all three notes and may
    # correct them. Applying the prepared mask cannot recompute their endings.
    for note in notes:
        note.end += .02
    assert proposal.apply(midi) == 1
    assert parts[0].notes[0] is notes[0] and parts[1].notes[0] is notes[2]
    assert [(n.start, n.end, n.pitch, n.velocity) for p in parts for n in p.notes] == [
        (3., 4.02, 48, 80), (4.04, 5.02, 48, 70)]


def test_stale_proposal_cannot_delete_from_changed_input():
    note = SimpleNamespace(start=3., end=4., pitch=48, velocity=80)
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[note])])
    proposal = service.RemovalProposal((id(note),), frozenset())
    midi.instruments[0].notes.append(SimpleNamespace(start=5., end=6., pitch=48, velocity=80))
    with pytest.raises(PipelineError, match="no longer match"):
        proposal.apply(midi)
    assert len(midi.instruments[0].notes) == 2


def test_empty_input_needs_no_audio_or_network():
    verifier = service.LeftHandPresence.__new__(service.LeftHandPresence)
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[])])
    assert verifier.prepare("missing.wav", {}, midi).apply(midi) == 0


def test_nonfinite_evidence_cannot_propose_any_deletion():
    with pytest.raises(ValueError, match="probabilities"):
        gated((np.array([[0., 1.]]), np.full((1, 40), np.nan)), .2)
