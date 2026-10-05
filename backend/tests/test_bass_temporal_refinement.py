"""V16 requires sealed arrays and independent KEEP vetoes before deletion."""

from types import SimpleNamespace

import numpy as np
import pytest
from app.models import PipelineError
from app.services import bass_temporal as inherited
from app.services import bass_temporal_refinement as service


def test_sealed_arrays_load_and_corruption_is_rejected(tmp_path, monkeypatch):
    verifier = service.BassTemporalRefinement()
    assert verifier.model.threshold == .2 and verifier.guardian.threshold == .3
    broken = tmp_path / 'broken.npz'
    broken.write_bytes(b'not an npz model')
    monkeypatch.setattr(service, 'ASSET', broken)
    with pytest.raises(PipelineError, match='missing or damaged'):
        service.BassTemporalRefinement()


@pytest.mark.parametrize('change', ['version', 'width', 'threshold', 'guardian_threshold', 'scale',
                                   'weight', 'weight_dtype', 'weight_shape', 'extra_array'])
def test_changed_portable_contract_is_rejected(tmp_path, change):
    with np.load(service.ASSET, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    key = next(key for key in arrays if key.startswith('weight_'))
    if change == 'scale':
        arrays['scale'][0] = 0
    elif change == 'weight':
        arrays[key].flat[0] = np.nan
    elif change == 'weight_dtype':
        arrays[key] = arrays[key].astype(np.float64)
    elif change == 'weight_shape':
        arrays[key] = arrays[key].ravel()
    elif change == 'extra_array':
        arrays['unexpected_model'] = np.zeros(1)
    else:
        arrays[change] = {'version': 'other-version', 'width': 32, 'threshold': .1, 'guardian_threshold': .4}[change]
    path = tmp_path / 'changed.npz'
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        service.RefinedTemporalCandidateModel(saved)


@pytest.mark.parametrize('first,second,removed', [(0., 0., 1), (.2, 0., 0), (0., .3, 0), (1., 0., 0), (0., 1., 0)])
def test_refinement_preserves_vetoes_holds_and_part_identity(monkeypatch, first, second, removed):
    verifier = service.BassTemporalRefinement.__new__(service.BassTemporalRefinement)
    verifier.model = SimpleNamespace(threshold=.2, probability=lambda frames, x: np.full(len(x), first))
    verifier.guardian = SimpleNamespace(threshold=.3, probability=lambda x: np.full(len(x), second))
    notes = [SimpleNamespace(start=3., end=14., pitch=40, velocity=90),
             SimpleNamespace(start=3., end=14., pitch=60, velocity=80)]
    original = [vars(n).copy() for n in notes]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[n]) for n in notes])
    monkeypatch.setattr(inherited.sf, 'read', lambda *a, **kw: (np.zeros(30 * 22050, np.float32), 22050))
    monkeypatch.setattr(inherited, 'note_features', lambda *a, **kw: np.zeros((2, 52), np.float32))
    monkeypatch.setattr(inherited, 'spectrum', lambda *a: None)
    monkeypatch.setattr(inherited, 'sequences', lambda *a: np.zeros((2, 40, 9), np.float32))
    assert verifier.filter('fixture.wav', {}, midi) == removed
    assert midi.instruments[0].notes == ([] if removed else [notes[0]])
    assert midi.instruments[1].notes == [notes[1]]
    assert [vars(n) for n in notes] == original
