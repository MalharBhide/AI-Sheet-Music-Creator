"""Training and context contracts independent of fitted datasets."""

import numpy as np
import pytest
import torch
from cache_vocalset import annotation_clock_scale
from melody_candidate import ContextMelodyDecoder
from train_context_melody import batch, loss

from app.services.melody_decoder import MelodyDecoder


def test_dataset_clock_correction_is_metadata_based_and_rejects_unknown_rates():
    assert annotation_clock_scale(3.838, 7.674) == .5
    assert annotation_clock_scale(9.94, 9.938) == 1
    with pytest.raises(ValueError):
        annotation_clock_scale(7, 9)


@pytest.mark.parametrize('frames', [1, 23, 1601])
def test_context_candidate_starts_identical_to_released_architecture(frames):
    torch.set_num_threads(2)
    original = MelodyDecoder().eval()
    candidate = ContextMelodyDecoder().eval()
    candidate.load_state_dict(original.state_dict(), strict=False)
    x = torch.randn(1, frames, 88, 10)
    with torch.inference_mode():
        expected = original(x)
        actual = candidate(x)
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_short_training_clip_is_padded_without_teaching_false_silence():
    torch.set_num_threads(2)
    item = {'x': np.ones((17, 88, 10), np.float32), 'y': np.full(17, 39, np.int64),
            'attacks': np.zeros((17, 88), np.float32)}
    item['attacks'][0, 39] = 1
    x, y, attacks = batch([[item]], np.random.default_rng(0))
    assert torch.all(y[:, :17] == 39)
    assert torch.all(y[:, 17:] == -100)
    assert not torch.any(x[:, 17:]) and not torch.any(attacks[:, 17:])
    model = ContextMelodyDecoder()
    value = loss(model, x, y, attacks)
    value.backward()
    assert torch.isfinite(value)
    assert model.context[-1].weight.grad.abs().sum() > 0


def test_loss_ignores_model_predictions_in_padded_frames():
    class Output:
        def __init__(self, pad):
            self.pad = pad

        def __call__(self, x):
            logits, attacks = torch.zeros(1, 8, 89), torch.zeros(1, 8, 88)
            logits[:, 3:, 0] = self.pad
            attacks[:, 3:] = self.pad
            return logits, attacks

    target = torch.tensor([[39, 39, 88, -100, -100, -100, -100, -100]])
    attacks = torch.zeros(1, 8, 88)
    assert loss(Output(0), None, target, attacks) == loss(Output(100), None, target, attacks)
