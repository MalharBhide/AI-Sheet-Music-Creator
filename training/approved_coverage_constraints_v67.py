"""Protect supplied pitch coverage as well as labeled KEEP note decisions."""

import torch

from approved_readout_constraints_v66 import training_constraints


def coverage_targets(note_target, frame_target, frame_weight):
    if (note_target.ndim != 1 or note_target.dtype != torch.int64
            or frame_target.shape != (len(note_target), 40) or frame_weight.shape != frame_target.shape
            or any(v.requires_grad for v in (note_target, frame_target, frame_weight))
            or not torch.isfinite(frame_target).all() or not torch.isfinite(frame_weight).all()
            or torch.any((note_target != 0) & (note_target != 1))
            or torch.any((frame_target < 0) | (frame_target > 1))
            or torch.any((frame_weight != 0) & (frame_weight != 1))):
        raise ValueError("Invalid fitting-only pitch coverage labels")
    known_support = ((frame_target[:, 8:32] > 0) & (frame_weight[:, 8:32] == 1)).any(dim=1)
    protected = (note_target == 0) | known_support
    return (~protected).long()


def coverage_constraints(network, inputs, note_target, frame_target, frame_weight, retention):
    # This mask governs teacher-decision protection only. Original note labels,
    # supervised frame targets and UNKNOWN masks remain unchanged for fitting.
    protection = coverage_targets(note_target, frame_target, frame_weight)
    return training_constraints(network, inputs, protection, retention)
