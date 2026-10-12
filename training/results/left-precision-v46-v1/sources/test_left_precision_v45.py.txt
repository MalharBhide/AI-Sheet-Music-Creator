import numpy as np
import pytest
import torch
from left_precision_v45 import frame_supervision, masked_frame_loss, pitch_evaluate, rehearsal_loss


def test_unknown_frames_have_no_loss_or_gradient():
    logits = torch.zeros((2, 40), requires_grad=True)
    targets = torch.ones_like(logits)
    mask = torch.zeros_like(logits)
    mask[0, 0] = 1
    result = masked_frame_loss(logits, targets, mask, None)
    assert result[1] == 0
    result.sum().backward()
    assert logits.grad[0, 0] != 0
    assert torch.count_nonzero(logits.grad) == 1
    targets[mask == 0] = 0
    assert torch.equal(masked_frame_loss(logits, targets, mask, None), result)


def test_rehearsal_loss_and_gradient_zero_at_original_prediction():
    rejection = torch.tensor([[2., -1.]], requires_grad=True)
    presence = torch.linspace(-2., 2., 40).reshape(1, 40).requires_grad_()
    teacher_r, teacher_p = rejection.detach().softmax(1), presence.detach().sigmoid()
    mask = torch.ones_like(presence)
    mask[:, 8:32] = 0
    loss = rehearsal_loss(rejection, presence, teacher_r, teacher_p, mask).sum()
    assert abs(float(loss)) < 1e-6
    loss.backward()
    assert float(rejection.grad.abs().max()) < 1e-6
    assert float(presence.grad.abs().max()) < 1e-6


def weak_item():
    return {"id": "isolated", "corpus": "tinysol-isolated-pitch", "source_group": "held-out-cello",
        "events": np.array([[3., 4., 40., 80.], [3.2, 4., 52., 80.]]),
        "duration": 8., "y": np.array([1., 0.]), "timing_supervision_available": False,
        "frame_target": np.zeros((2, 40), np.float32), "frame_weight": np.ones((2, 40), np.float32)}


def baseline(item, accepted=(True, True)):
    previous = np.array(accepted)
    return {**item, "events": item["events"][previous].copy(), "baseline_accepted": previous,
            "full_release_events": item["events"].copy()}


def test_weak_pitch_validation_accepts_wrong_octave_removal_without_timing_claim():
    item = weak_item()
    report = pitch_evaluate([item], [baseline(item)], [np.array([[1., 0.], [0., 1.]])], .9)
    assert report["passes"] and report["wrong_pitch_gain"] == 1
    assert report["timing_supervision_available"] is False
    assert "matches" not in report["per_recording"][0]


def test_weak_pitch_validation_fails_lost_fundamental_or_restored_wrong_note():
    item = weak_item()
    lost = pitch_evaluate([item], [baseline(item)], [np.array([[0., 1.], [0., 1.]])], .9)
    assert not lost["passes"] and lost["removed_fundamental_candidates"] == 1
    restored = pitch_evaluate([item], [baseline(item, (True, False))], [np.ones((2, 2))*.5], .9)
    assert not restored["passes"] and restored["wrong_pitch_gain"] == -1


def test_invalid_unknown_mask_cannot_become_supervision():
    item = weak_item()
    item["frame_weight"][0, 0] = .1
    with pytest.raises(ValueError, match="masked"):
        frame_supervision(item)
    item["corpus"] = "unlabeled-recording"
    with pytest.raises(ValueError, match="corpus"):
        frame_supervision(item)
