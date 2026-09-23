"""Runtime safety and event preservation for the trained accompaniment model."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.models import PipelineError
from app.services import accompaniment_verifier as service
from app.services.note_context_model import correction_keep, residual_keep, shared_events


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
    monkeypatch.setattr(service, 'note_features', lambda *args, **kwargs: np.zeros((4, 52), dtype=np.float32))
    monkeypatch.setattr(service, 'decode_candidates', lambda _: midi)
    verifier.context_models = [SimpleNamespace(threshold=model.threshold, probability=lambda x: np.ones(len(x)))
                               for model in verifier.context_models]
    verifier.residual_model = SimpleNamespace(threshold=.0375, probability=lambda x: np.ones(len(x)))
    verifier.model = lambda x: torch.tensor([2., -5., 1., -4.])
    assert verifier.filter(path, {}, midi) == 2
    assert midi.instruments[0].notes == [notes[0]]
    assert midi.instruments[1].notes == [notes[2]]
    assert [vars(n) for n in notes] == before


def test_context_correction_matches_runtime_without_changing_unshared_or_confident_notes(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    path = tmp_path / 'window.wav'
    sf.write(path, np.zeros(22050), 22050)
    notes = [SimpleNamespace(pitch=60 + i, start=.1 * i, end=.5 + .1 * i, velocity=70 + i)
             for i in range(8)]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes[:4]), SimpleNamespace(notes=notes[4:])])
    before = [vars(n).copy() for n in notes]
    # Decoder order is unrelated to instrument order. The almost identical
    # fifth candidate differs in velocity and must not receive a correction.
    bounded_notes = [SimpleNamespace(**before[i]) for i in [7, 3, 1, 6, 0, 2, 5]]
    bounded_notes.append(SimpleNamespace(**{**before[4], 'velocity': notes[4].velocity + 1}))
    bounded = SimpleNamespace(instruments=[SimpleNamespace(notes=bounded_notes)])
    monkeypatch.setattr(service, 'decode_candidates', lambda _: bounded)
    x = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 100

    def features(samples, rate, acoustic, candidates, *, include_context=False):
        assert include_context and candidates == notes and rate == 22050
        return x

    monkeypatch.setattr(service, 'note_features', features)
    previous = np.array([.8, .15, .15, .15, .15, .05, .2, .2001])
    prune = verifier.context_models[0].threshold
    low, high = prune / 5, min(1., prune * 2)
    context = np.array([[low, low, low, high, low, .8, low, low],
                        [low, low, high, low, low, .8, low, low]])

    def old_model(features):
        np.testing.assert_array_equal(features.numpy(), (x[:, :26] - verifier.mean) / verifier.scale)
        return torch.from_numpy(np.log(previous / (1 - previous)))

    def context_model(index):
        def probability(features):
            np.testing.assert_array_equal(features, x)
            return context[index]
        return SimpleNamespace(threshold=prune, probability=probability)

    verifier.model = old_model
    verifier.context_models = [context_model(0), context_model(1)]
    verifier.residual_model = SimpleNamespace(threshold=.0375, probability=lambda x: np.ones(len(x)))
    events = [[n.start, n.end, n.pitch, n.velocity] for n in notes]
    bounded_events = [[n.start, n.end, n.pitch, n.velocity] for n in bounded_notes]
    shared = shared_events(events, bounded_events)
    np.testing.assert_array_equal(shared, [True, True, True, True, False, True, True, True])
    expected = correction_keep(previous, context, shared, prune=prune)
    np.testing.assert_array_equal(expected, [True, False, True, True, True, False, False, True])
    assert verifier.filter(path, {}, midi) == 3
    assert midi.instruments[0].notes == [notes[0], notes[2], notes[3]]
    assert midi.instruments[1].notes == [notes[4], notes[7]]
    assert [n for part in midi.instruments for n in part.notes] == [n for n, keep in zip(notes, expected, strict=True) if keep]
    assert [vars(n) for n in notes] == before


def test_residual_model_matches_evaluation_and_preserves_note_objects(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    assert 'v6' in verifier.name
    path = tmp_path / 'window.wav'
    sf.write(path, np.zeros(22050), 22050)
    notes = [SimpleNamespace(pitch=60 + i, start=i * .1, end=1. + i * .1, velocity=80)
             for i in range(7)]
    original = [vars(n).copy() for n in notes]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes[:3]), SimpleNamespace(notes=notes[3:])])
    bounded = SimpleNamespace(instruments=[SimpleNamespace(notes=[notes[i] for i in [6, 5, 4, 3, 1, 0]])])
    monkeypatch.setattr(service, 'decode_candidates', lambda _: bounded)
    x = np.arange(7 * 52, dtype=np.float32).reshape(7, 52) / 100
    monkeypatch.setattr(service, 'note_features', lambda *args, **kwargs: x)
    p = np.array([.2, .5001, .2, .05, .4, .5, .3])
    verifier.model = lambda _: torch.from_numpy(np.log(p / (1 - p)))
    verifier.context_models = [SimpleNamespace(threshold=.01, probability=lambda x: np.ones(len(x)))] * 2

    def residual_probability(features):
        # Confident/unshared/already-rejected events do not need another model pass.
        np.testing.assert_array_equal(features, x[[0, 4, 5, 6]])
        return np.array([.001, .0375, .001, .9])

    verifier.residual_model = SimpleNamespace(threshold=.0375, probability=residual_probability)
    assert verifier.filter(path, {}, midi) == 3
    assert midi.instruments[0].notes == [notes[1], notes[2]]
    assert midi.instruments[1].notes == [notes[4], notes[6]]
    assert [vars(n) for n in notes] == original


def test_residual_confidence_rejects_invalid_scores():
    for invalid in (np.array([np.nan]), np.array([1.01]), np.array([[.1]])):
        with pytest.raises(ValueError, match='residual'):
            residual_keep(np.array([True]), np.array([.2]), np.array([True]), invalid, threshold=.0375)


def test_damaged_residual_model_fails_explicitly(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'residual.npz'
    path.write_bytes(b'invalid model')
    monkeypatch.setattr(service, 'RESIDUAL_CHECKPOINT', path)
    with pytest.raises(PipelineError, match='residual note model is missing or damaged'):
        service.AccompanimentVerifier()


def test_low_register_specialist_preserves_protected_notes_and_all_attributes(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    path = tmp_path / 'window.wav'
    sf.write(path, np.zeros(22050), 22050)
    notes = [SimpleNamespace(pitch=p, start=i * .1, end=2. + i * .1, velocity=80)
             for i, p in enumerate([48, 50, 52, 54, 55, 59, 60, 72])]
    before = [vars(n).copy() for n in notes]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes[:4]), SimpleNamespace(notes=notes[4:])])
    bounded = SimpleNamespace(instruments=[SimpleNamespace(notes=[n for i, n in enumerate(notes) if i != 2])])
    monkeypatch.setattr(service, 'decode_candidates', lambda _: bounded)
    x = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 100
    monkeypatch.setattr(service, 'note_features', lambda *args, **kwargs: x)
    p = np.array([.2, .5001, .2, .05, .4, .5, .3, .2])
    verifier.model = lambda _: torch.from_numpy(np.log(p / (1 - p)))
    verifier.context_models = [SimpleNamespace(threshold=.01, probability=lambda x: np.ones(len(x)))] * 2
    verifier.residual_model = SimpleNamespace(threshold=.0375, probability=lambda x: np.ones(len(x)))

    def low_probability(features):
        np.testing.assert_array_equal(features, x[[0, 4, 5]])
        return np.array([.001, .05, .001])

    verifier.left_hand_model = SimpleNamespace(threshold=.05, probability=low_probability)
    verifier.left_refinement_model = SimpleNamespace(threshold=.0375, probability=lambda x: np.ones(len(x)))
    assert verifier.filter(path, {}, midi) == 3
    assert midi.instruments[0].notes == [notes[1], notes[2]]
    assert midi.instruments[1].notes == [notes[4], notes[6], notes[7]]
    assert [vars(n) for n in notes] == before


def test_damaged_left_hand_model_fails_explicitly(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'left-hand.npz'
    path.write_bytes(b'invalid model')
    monkeypatch.setattr(service, 'LEFT_HAND_CHECKPOINT', path)
    with pytest.raises(PipelineError, match='left-hand note model is missing or damaged'):
        service.AccompanimentVerifier()


def test_refinement_only_scores_remaining_uncertain_low_notes(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    path = tmp_path / 'window.wav'
    sf.write(path, np.zeros(22050), 22050)
    notes = [SimpleNamespace(pitch=p, start=i * .1, end=2. + i * .1, velocity=80)
             for i, p in enumerate([48, 50, 52, 54, 55, 59, 60, 72])]
    before = [vars(n).copy() for n in notes]
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=notes[:4]), SimpleNamespace(notes=notes[4:])])
    bounded = SimpleNamespace(instruments=[SimpleNamespace(notes=[n for i, n in enumerate(notes) if i != 2])])
    monkeypatch.setattr(service, 'decode_candidates', lambda _: bounded)
    x = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 100
    monkeypatch.setattr(service, 'note_features', lambda *args, **kwargs: x)
    p = np.array([.2, .5001, .2, .05, .4, .5, .3, .2])
    verifier.model = lambda _: torch.from_numpy(np.log(p / (1 - p)))
    verifier.context_models = [SimpleNamespace(threshold=.01, probability=lambda x: np.ones(len(x)))] * 2
    verifier.residual_model = SimpleNamespace(threshold=.0375, probability=lambda x: np.ones(len(x)))
    verifier.left_hand_model = SimpleNamespace(threshold=.05, probability=lambda x: np.array([1., 1., .001]))

    def refinement(features):
        np.testing.assert_array_equal(features, x[[0, 4]])
        return np.array([.001, .0375])

    verifier.left_refinement_model = SimpleNamespace(threshold=.0375, probability=refinement)
    assert verifier.filter(path, {}, midi) == 3
    assert midi.instruments[0].notes == [notes[1], notes[2]]
    assert midi.instruments[1].notes == [notes[4], notes[6], notes[7]]
    assert [vars(n) for n in notes] == before


def test_damaged_refinement_model_fails_explicitly(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'refinement.npz'
    path.write_bytes(b'invalid model')
    monkeypatch.setattr(service, 'LEFT_REFINEMENT_CHECKPOINT', path)
    with pytest.raises(PipelineError, match='refinement model is missing or damaged'):
        service.AccompanimentVerifier()


@pytest.mark.parametrize('damaged_index', [0, 1])
def test_damaged_context_checkpoint_fails_explicitly(monkeypatch, tmp_path, damaged_index):
    pytest.importorskip('torch')
    damaged = tmp_path / 'damaged.npz'
    damaged.write_bytes(b'not the checked context model')
    checkpoints = list(service.CONTEXT_CHECKPOINTS)
    checkpoints[damaged_index] = (damaged, checkpoints[damaged_index][1])
    monkeypatch.setattr(service, 'CONTEXT_CHECKPOINTS', tuple(checkpoints))
    with pytest.raises(PipelineError, match='context model is missing or damaged'):
        service.AccompanimentVerifier()


def test_rejects_unbounded_and_non_normalized_audio(tmp_path):
    pytest.importorskip('torch')
    verifier = service.AccompanimentVerifier()
    midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[SimpleNamespace()])])
    for rate, shape in [(16000, (16000,)), (22050, (22050, 2)), (22050, (34 * 22050,))]:
        path = tmp_path / 'window.wav'
        sf.write(path, np.zeros(shape), rate)
        with pytest.raises(PipelineError, match='normalized window'):
            verifier.filter(path, {}, midi)
