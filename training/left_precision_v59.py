"""Align fitting absence loss with the immutable maximum-interior-frame veto."""

from left_precision_v58 import (
    frame_supervision,
    gate,
    hard_keep_loss,
    isolated_negative_weights,
    model,
    probabilities,
    real_piano_weights,
    require_phase_contract,
    validate_partitions,
)

VERSION = "known-absent-interior-maximum-loss-v59-v1"
FRESH_SEEDS = tuple(range(303101, 303141))
__all__ = ["VERSION", "FRESH_SEEDS", "model", "gate", "probabilities", "frame_supervision",
           "hard_keep_loss", "isolated_negative_weights", "real_piano_weights",
           "require_phase_contract", "validate_partitions", "absence_candidates", "max_absence_loss"]


def absence_candidates(target, frame_weight, note_target):
    """Only supervised REMOVE notes with every interior frame known absent."""
    import torch

    if (target.ndim != 2 or target.shape[1] != 40 or frame_weight.shape != target.shape
            or note_target.shape != target.shape[:1] or note_target.dtype != torch.int64
            or any(value.requires_grad for value in (target, frame_weight, note_target))
            or not torch.isfinite(target).all() or not torch.isfinite(frame_weight).all()
            or torch.any((target < 0) | (target > 1))
            or torch.any((frame_weight != 0) & (frame_weight != 1))
            or torch.any((note_target != 0) & (note_target != 1))):
        raise ValueError("Invalid known-absence fitting supervision")
    return ((note_target == 1) & (target[:, 8:32] == 0).all(dim=1)
            & (frame_weight[:, 8:32] == 1).all(dim=1))


def max_absence_loss(logits, target, frame_weight, note_target, weights):
    """Penalize the strongest false support frame, without changing inference.

    Genuine KEEP notes, any interior UNKNOWN frame and even a single supplied
    positive fragment are excluded. The penalty uses fitting supervision only.
    """
    import torch

    selected = absence_candidates(target, frame_weight, note_target)
    if (logits.shape != target.shape or not torch.isfinite(logits).all()
            or weights.shape != note_target.shape or weights.requires_grad
            or not torch.isfinite(weights).all() or torch.any(weights <= 0)):
        raise ValueError("Invalid maximum-absence fitting batch")
    if not selected.any():
        # Empty reduction preserves a zero gradient without overflowing a sum
        # of otherwise finite but large excluded logits.
        return logits.reshape(-1)[:0].sum()
    maximum = logits[selected, 8:32].amax(dim=1)
    return (torch.nn.functional.softplus(maximum)*weights[selected]).mean()
