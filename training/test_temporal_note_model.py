"""Temporal evidence must preserve time, pitch relativity and silent boundaries."""

import numpy as np
import pytest
from app.services.temporal_note_model import (
    FEATURE_COUNT,
    OFFSETS,
    RATE,
    STEPS,
    create_model,
    probability,
    sequences,
    spectrum,
)


def test_phase_grid_tracks_actual_attack_and_release():
    from app.services.temporal_note_model import HOP

    clock = np.arange(600) * HOP / RATE
    observed = np.tile(clock[:, None] / 10, (1, 88)).astype(np.float32)
    events = np.array([[3., 4., 60., 90.]])
    actual = sequences(observed, events)
    expected = np.concatenate((3 - np.linspace(.24, .03, 8), np.linspace(3, 4, 24), 4 + np.linspace(.03, .24, 8))) / 10
    np.testing.assert_allclose(actual[0, :, 0], expected, atol=1e-7)
    assert actual.shape == (1, STEPS, len(OFFSETS))
    assert actual[0, 8, 0] == pytest.approx(.3) and actual[0, 31, 0] == pytest.approx(.4)


def test_observed_channels_are_pitch_relative_and_empty_candidates_are_valid():
    rng = np.random.default_rng(123)
    observed = rng.random((600, 88), dtype=np.float32)
    events = np.array([[3., 4., 60., 90.]])
    expected = sequences(observed, events)
    events[:, 2] += 3
    np.testing.assert_array_equal(expected, sequences(np.roll(observed, 3, axis=1), events))
    assert sequences(observed, np.empty((0, 4))).shape == (0, STEPS, len(OFFSETS))


def test_silence_and_short_audio_do_not_become_full_strength_spectra():
    pytest.importorskip('librosa')
    observed = spectrum(np.zeros(RATE // 2, np.float32), RATE)
    assert observed.shape[1] == 88 and not observed.any()


@pytest.mark.parametrize('audio,rate', [(np.zeros(4), 44100), (np.zeros((4, 2)), RATE), (np.full(4, np.nan), RATE), (np.zeros(0), RATE)])
def test_invalid_audio_is_rejected(audio, rate):
    with pytest.raises(ValueError, match='finite mono'):
        spectrum(audio, rate)


def test_neural_probability_supports_empty_and_bounded_batches():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    torch.manual_seed(33)
    model = create_model(16)
    frames, context = np.zeros((3, STEPS, len(OFFSETS)), np.float32), np.zeros((3, FEATURE_COUNT), np.float32)
    normalizer = (np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT))
    first = probability(model, frames, context, normalizer, 2)
    second = probability(model, frames, context, normalizer, 3)
    np.testing.assert_allclose(first, second, atol=1e-7)
    assert ((first > 0) & (first < 1)).all()
    assert probability(model, frames[:0], context[:0], normalizer).shape == (0,)
    with pytest.raises(ValueError, match='inputs'):
        probability(model, frames, context, (np.zeros(FEATURE_COUNT), np.zeros(FEATURE_COUNT)))


def test_supervised_temporal_rows_follow_context_corpus_order_and_masks():
    from train_bass_temporal import supervised_arrays

    def item(corpus, marker, label):
        return {'id': corpus, 'corpus': corpus, 'x': np.full((2, FEATURE_COUNT), marker),
                'frames': np.full((2, STEPS, len(OFFSETS)), marker), 'y': np.full(2, label),
                'eligible': np.array([True, False]), 'mask': np.ones(2, bool)}
    x, frames, y, _, counts = supervised_arrays([item('zeta', .8, 0.), item('alpha', .2, 1.)], 12.)
    np.testing.assert_array_equal(x[:, 0], frames[:, 0, 0])
    np.testing.assert_array_equal(y, [1., 0.])
    assert [r['id'] for r in counts] == ['alpha', 'zeta']
