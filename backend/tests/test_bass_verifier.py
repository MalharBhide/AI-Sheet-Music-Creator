from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import bass_verifier as service


def test_consensus_preserves_holds_equal_thresholds_edges_and_other_registers(monkeypatch):
    notes = [SimpleNamespace(start=s, end=e, pitch=p, velocity=90, source='bass') for s, e, p in
             [(3., 11., 36), (12., 13., 40), (14., 15., 52), (.1, 2., 40), (25., 29., 40), (20., 21., 60)]]
    attributes = [vars(note).copy() for note in notes]
    parts = [SimpleNamespace(notes=notes[:3].copy()), SimpleNamespace(notes=notes[3:].copy())]
    verifier = service.BassVerifier()
    p, g = np.array([0., .1, 0., 0., 0., 0.]), np.array([.2, 0., 0., 0., 0., 0.])
    verifier.models = [SimpleNamespace(feature_count=52, threshold=.1, probability=lambda x: p),
                       SimpleNamespace(feature_count=26, threshold=.1, probability=lambda x: g)]
    monkeypatch.setattr(service.sf, 'read', lambda *args, **kwargs: (np.zeros(30 * 22050), 22050))
    monkeypatch.setattr(service, 'note_features', lambda *args, **kwargs: np.zeros((6, 52), dtype=np.float32))
    assert verifier.filter('cached.wav', {}, SimpleNamespace(instruments=parts)) == 1
    assert parts[0].notes == notes[:2] and parts[1].notes == notes[3:]
    assert all(actual is expected for actual, expected in zip(parts[0].notes, notes[:2], strict=True))
    assert [vars(note) for note in notes] == attributes


def test_empty_bass_stem_never_reads_audio_or_predicts():
    verifier = service.BassVerifier()
    assert verifier.filter('does-not-exist.wav', {}, SimpleNamespace(instruments=[])) == 0


def test_damaged_or_missing_bass_assets_fail_explicitly(tmp_path, monkeypatch):
    path = tmp_path / 'damaged.npz'
    monkeypatch.setattr(service, 'CHECKPOINTS', [(path, '0' * 64, ())])
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.BassVerifier()
    path.write_bytes(b'not-model-arrays')
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.BassVerifier()


@pytest.mark.parametrize('samples,rate', [(np.zeros((100, 2)), 22050), (np.zeros(100), 44100),
                                        (np.full(100, np.nan), 22050)])
def test_bass_waveform_contract_rejects_stereo_wrong_rate_and_nonfinite_audio(monkeypatch, samples, rate):
    verifier = service.BassVerifier()
    note = SimpleNamespace(start=3., end=4., pitch=40, velocity=90)
    monkeypatch.setattr(service.sf, 'read', lambda *args, **kwargs: (samples, rate))
    with pytest.raises(PipelineError, match='finite mono'):
        verifier.filter('fixture.wav', {}, SimpleNamespace(instruments=[SimpleNamespace(notes=[note])]))
