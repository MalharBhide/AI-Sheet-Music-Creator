import pytest
import torch

from approved_coverage_constraints_v67 import coverage_targets


def test_wrong_attack_with_known_pitch_support_remains_protected():
    notes = torch.tensor([0, 1, 1, 1, 1])
    frames = torch.zeros(5, 40)
    mask = torch.ones_like(frames)
    frames[1, 12] = 1  # Wrong attack, real interior pitch coverage.
    frames[2, 12] = 1
    mask[2, 12] = 0  # Unknown command support is not acoustic evidence.
    frames[3, 3] = 1  # Support only outside the candidate interior.
    before = (notes.clone(), frames.clone(), mask.clone())
    assert coverage_targets(notes, frames, mask).tolist() == [0, 0, 1, 1, 1]
    for value, original in zip((notes, frames, mask), before, strict=True):
        assert torch.equal(value, original)


@pytest.mark.parametrize('fault', ['nan', 'mask', 'label', 'shape', 'grad'])
def test_invalid_or_trainable_supervision_is_rejected(fault):
    notes = torch.tensor([0, 1])
    frames = torch.zeros(2, 40)
    mask = torch.ones_like(frames)
    if fault == 'nan': frames[0, 0] = float('nan')
    if fault == 'mask': mask[0, 0] = .5
    if fault == 'label': notes[0] = 2
    if fault == 'shape': frames = frames[:, :39]
    if fault == 'grad': frames.requires_grad_()
    with pytest.raises(ValueError, match='coverage labels'):
        coverage_targets(notes, frames, mask)
