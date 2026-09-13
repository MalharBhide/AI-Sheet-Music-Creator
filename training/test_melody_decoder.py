"""Contract tests independent of training recordings or fitted weights."""

import numpy as np
import pytest
import torch

from app.services.melody_decoder import CHANNELS, MelodyDecoder, decode, labels


def evidence(pitches):
    logits = np.full((len(pitches), 89), -10.)
    logits[np.arange(len(pitches)), np.asarray(pitches)] = 10
    return logits, np.zeros((len(pitches), 88))


def test_labels_read_hz_and_duration_without_stretching_a_note():
    target, attacks = labels(np.array([[.5, 440., .3]]), 100)
    assert np.flatnonzero(target != 88).tolist() == list(range(25, 40))
    assert np.all(target[25:40] == 48)
    assert np.flatnonzero(attacks[:, 48]).tolist() == [24, 25, 26]


def test_silence_stays_silent_and_a_real_rest_is_preserved():
    assert decode(np.empty((0, 89)), np.empty((0, 88))).shape == (0, 3)
    assert decode(*evidence([88] * 50)).shape == (0, 3)
    result = decode(*evidence([39] * 25 + [88] * 50 + [43] * 25))
    assert result.tolist() == [[0., .5, 60.], [1.5, 2., 64.]]


def test_octave_jump_does_not_invent_an_intermediate_pitch():
    result = decode(*evidence([36] * 25 + [48] * 25), smoothing=5)
    assert result[:, 2].tolist() == [57, 69]


def test_real_repeat_is_split_but_a_held_note_is_not():
    logits, attacks = evidence([39] * 100)
    assert decode(logits, attacks).tolist() == [[0., 2., 60.]]
    attacks[50, 39] = .95
    assert decode(logits, attacks).tolist() == [[0., 1., 60.], [1., 2., 60.]]


def test_brief_pitch_glitch_is_not_a_new_note():
    result = decode(*evidence([39] * 30 + [51] + [39] * 30), smoothing=3)
    assert result.tolist() == [[0., 1.22, 60.]]


@pytest.mark.parametrize('frames', [1, 17, 1601])
def test_model_handles_short_and_long_windows_without_nonfinite_values(frames):
    torch.set_num_threads(2)
    model = MelodyDecoder().eval()
    with torch.no_grad():
        logits, attacks = model(torch.zeros(1, frames, 88, CHANNELS))
    assert logits.shape == (1, frames, 89)
    assert attacks.shape == (1, frames, 88)
    assert torch.isfinite(logits).all() and torch.isfinite(attacks).all()
