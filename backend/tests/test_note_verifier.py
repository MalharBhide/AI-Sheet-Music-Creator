"""Runtime safety and event preservation for the trained accompaniment model."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.models import PipelineError
from app.services import accompaniment_verifier as service


def test_missing_checkpoint_fails_explicitly(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    monkeypatch.setattr(service, 'CHECKPOINT', tmp_path / 'missing.pt')
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.AccompanimentVerifier()


def test_filter_preserves_retained_notes_and_instrument_assignment(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    assert verifier.threshold == .1
    path = tmp_path / 'window.wav'
    sf.write(path, np.zeros(22050), 22050)
    notes = [SimpleNamespace(pitch=p, start=.1 * i, end=.5 + .1 * i, velocity=60 + i)
             for i, p in enumerate([60, 72, 64, 79])]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes[:2]), SimpleNamespace(notes=notes[2:])])
    before = [vars(n).copy() for n in notes]
    monkeypatch.setattr(service, 'note_features', lambda *args: np.zeros((4, 26), dtype=np.float32))
    verifier.model = lambda x: torch.tensor([2., -5., 1., -4.])
    assert verifier.filter(path, {}, midi) == 2
    assert midi.instruments[0].notes == [notes[0]]
    assert midi.instruments[1].notes == [notes[2]]
    assert [vars(n) for n in notes] == before


def test_rejects_unbounded_and_non_normalized_audio(tmp_path):
    pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[SimpleNamespace()])])
    for rate, shape in [(16000, (16000,)), (22050, (22050, 2)), (22050, (34 * 22050,))]:
        path = tmp_path / 'window.wav'
        sf.write(path, np.zeros(shape), rate)
        with pytest.raises(PipelineError, match='normalized window'):
            verifier.filter(path, {}, midi)
