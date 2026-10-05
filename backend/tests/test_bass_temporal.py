"""Unsupported-pitch rejection must protect vetoes, timing and part identity."""

from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import bass_temporal as service


def setup(monkeypatch, first=0., second=0.):
    verifier = service.BassTemporal.__new__(service.BassTemporal)
    verifier.model = SimpleNamespace(threshold=.005, probability=lambda frames, x: np.full(len(x), first))
    verifier.guardian = SimpleNamespace(threshold=.3, probability=lambda x: np.full(len(x), second))
    notes = [SimpleNamespace(start=3., end=4., pitch=40, velocity=80, origin=i) for i in range(2)]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[note]) for note in notes])
    monkeypatch.setattr(service.sf, 'read', lambda *a, **kw: (np.zeros(30 * 22050, np.float32), 22050))
    monkeypatch.setattr(service, 'note_features', lambda samples, rate, acoustic, notes, **kw: np.zeros((len(notes), 52), np.float32))
    monkeypatch.setattr(service, 'spectrum', lambda *a: None)
    monkeypatch.setattr(service, 'sequences', lambda observed, events: np.zeros((len(events), 40, 9), np.float32))
    return verifier, midi, notes


def test_only_both_heads_can_remove_notes_and_objects_are_not_changed(monkeypatch):
    verifier, midi, notes = setup(monkeypatch)
    original = [vars(n).copy() for n in notes]
    notes[1].pitch = 60
    original[1]['pitch'] = 60
    assert verifier.filter('fixture.wav', {}, midi) == 1
    assert not midi.instruments[0].notes
    assert midi.instruments[1].notes == [notes[1]]
    assert original == [vars(n) for n in notes]


@pytest.mark.parametrize('first,second', [(.005, 0.), (0., .3), (1., 0.), (0., 1.)])
def test_either_head_or_threshold_equality_preserves_notes(monkeypatch, first, second):
    verifier, midi, notes = setup(monkeypatch, first, second)
    assert verifier.filter('fixture.wav', {}, midi) == 0
    assert all(part.notes == [n] for part, n in zip(midi.instruments, notes, strict=True))


@pytest.mark.parametrize('start,end,pitch', [(2.49, 4., 40), (3., 27.51, 40), (3., 4., 20), (3., 4., 60)])
def test_window_edges_and_other_registers_are_protected(monkeypatch, start, end, pitch):
    verifier, midi, notes = setup(monkeypatch)
    for note in notes:
        note.start, note.end, note.pitch = start, end, pitch
    assert verifier.filter('fixture.wav', {}, midi) == 0


def test_empty_notes_skip_audio_and_invalid_audio_fails_clearly(monkeypatch):
    verifier, midi, _ = setup(monkeypatch)
    monkeypatch.setattr(service.sf, 'read', lambda *a, **kw: pytest.fail('Empty audio read'))
    assert verifier.filter('fixture.wav', {}, SimpleNamespace(instruments=[])) == 0
    for samples, rate in [(np.zeros(4), 44100), (np.zeros((4, 2)), 22050), (np.full(4, np.nan), 22050), (np.zeros(0), 22050)]:
        monkeypatch.setattr(service.sf, 'read', lambda *a, samples=samples, rate=rate, **kw: (samples, rate))
        with pytest.raises(PipelineError, match='finite mono'):
            verifier.filter('fixture.wav', {}, midi)


def test_released_arrays_load_and_corruption_is_rejected(tmp_path, monkeypatch):
    verifier = service.BassTemporal()
    assert verifier.model.threshold == .005 and verifier.guardian.threshold == .3
    damaged = tmp_path / 'damaged.npz'
    damaged.write_bytes(b'invalid')
    monkeypatch.setattr(service, 'ASSET', damaged)
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.BassTemporal()


@pytest.mark.parametrize('change', ['version', 'width', 'threshold', 'scale', 'weight'])
def test_incompatible_neural_arrays_are_rejected(tmp_path, change):
    with np.load(service.ASSET, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if change == 'scale':
        arrays['scale'][0] = 0.
    elif change == 'weight':
        key = next(key for key in arrays if key.startswith('weight_'))
        arrays[key].flat[0] = np.nan
    else:
        arrays[change] = {'version': 'wrong', 'width': 32, 'threshold': .01}[change]
    path = tmp_path / 'incompatible.npz'
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        service.TemporalCandidateModel(saved)


@pytest.mark.parametrize('evidence', [np.full((1, 52), np.nan), np.zeros((2, 52)), np.zeros((1, 51))])
def test_invalid_evidence_cannot_become_a_rejection(evidence):
    with pytest.raises(ValueError, match='evidence'):
        service.features(np.array([[3., 4., 40., 80.]]), evidence)
