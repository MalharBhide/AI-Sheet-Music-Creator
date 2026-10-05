"""Bass hold repair protects attacks, real rests, source parts and note identity."""

from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import bass_articulation as service


def setup(monkeypatch, first=0., second=0.):
    verifier = service.BassArticulation.__new__(service.BassArticulation)
    verifier.models = [SimpleNamespace(probability=lambda x, p=p: np.full(len(x), p)) for p in (first, second)]
    baseline = SimpleNamespace(models=[SimpleNamespace(feature_count=n, probability=lambda x: np.full(len(x), .9)) for n in (52, 26)])
    notes = [SimpleNamespace(start=float(3 + i), end=float(4 + i), pitch=40, velocity=80 - i, origin=i) for i in range(3)]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes.copy())])
    monkeypatch.setattr(service.sf, 'read', lambda *args, **kwargs: (np.zeros(30 * 22050, np.float32), 22050))
    monkeypatch.setattr(service, 'note_features', lambda samples, rate, acoustic, notes, **kwargs: np.zeros((len(notes), 52), np.float32))
    return verifier, baseline, midi, notes


def test_chain_keeps_earliest_note_identity_pitch_attack_and_velocity(monkeypatch):
    verifier, baseline, midi, notes = setup(monkeypatch)
    assert verifier.filter('fixture.wav', {}, midi, baseline) == 2
    assert midi.instruments[0].notes == [notes[0]]
    assert vars(notes[0]) == {'start': 3., 'end': 6., 'pitch': 40, 'velocity': 80, 'origin': 0}
    assert notes[1].end == 5. and notes[2].end == 6.


@pytest.mark.parametrize('first,second', [(.05, 0.), (0., .05), (1., 0.), (0., 1.)])
def test_either_head_and_threshold_equality_veto_merging(monkeypatch, first, second):
    verifier, baseline, midi, notes = setup(monkeypatch, first, second)
    assert verifier.filter('fixture.wav', {}, midi, baseline) == 0
    assert midi.instruments[0].notes == notes and notes[0].end == 4.


def test_real_rests_and_cross_instrument_boundaries_are_protected(monkeypatch):
    verifier, baseline, midi, notes = setup(monkeypatch)
    midi.instruments = [SimpleNamespace(notes=[note]) for note in notes]
    assert verifier.filter('fixture.wav', {}, midi, baseline) == 0
    assert all(part.notes == [note] for part, note in zip(midi.instruments, notes, strict=True))
    midi.instruments = [SimpleNamespace(notes=notes.copy())]
    notes[1].start += .01
    notes[2].start += .01
    assert verifier.filter('fixture.wav', {}, midi, baseline) == 0
    assert notes[0].end == 4.


def test_empty_notes_do_not_read_waveform_and_invalid_audio_cannot_merge(monkeypatch):
    verifier, baseline, midi, _ = setup(monkeypatch)
    monkeypatch.setattr(service.sf, 'read', lambda *a, **kw: pytest.fail('Empty waveform read'))
    assert verifier.filter('fixture.wav', {}, SimpleNamespace(instruments=[]), baseline) == 0
    for samples, rate in [(np.zeros(4), 44100), (np.zeros((4, 2)), 22050), (np.full(4, np.nan), 22050)]:
        monkeypatch.setattr(service.sf, 'read', lambda *a, samples=samples, rate=rate, **kw: (samples, rate))
        with pytest.raises(PipelineError, match='finite mono'):
            verifier.filter('fixture.wav', {}, midi, baseline)


def test_released_arrays_load_and_damaged_model_fails_clearly(tmp_path, monkeypatch):
    models = service.BassArticulation().models
    assert [m.feature_count for m in models] == [112, 60]
    assert [m.threshold for m in models] == [.05, .05]
    damaged = tmp_path / 'damaged.npz'
    damaged.write_bytes(b'invalid')
    monkeypatch.setattr(service, 'CHECKPOINTS', ((damaged, 'wrong-sha', service.RELATION_NAMES), service.CHECKPOINTS[1]))
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.BassArticulation()
