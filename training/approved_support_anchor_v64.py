"""Training-only protection of approved KEEP confidence and known pitch support."""

import math

import torch


def note_anchor(logits, target, teacher, weights):
    if (logits.shape != (len(target), 2) or target.dtype != torch.int64
            or teacher.shape != target.shape or weights.shape != target.shape
            or any(v.requires_grad for v in (target, teacher, weights))
            or not all(torch.isfinite(v).all() for v in (logits, teacher, weights))
            or torch.any((target != 0) & (target != 1)) or torch.any(weights <= 0)):
        raise ValueError("Invalid supervised note anchor")
    selected = target == 0
    if not selected.any():
        return logits.reshape(-1)[:0].sum()
    ceiling = teacher[selected].clamp(max=math.log(.88/.12))
    excess = (logits[selected, 1]-logits[selected, 0]-ceiling).clamp_min(0)
    return (excess.square()*weights[selected]).mean()


def support_anchor(logits, target, mask, note_target, teacher, weights):
    """Anchor only known positive interior frames; UNKNOWN has zero gradient.

    This protects supplied pitch support, not invented key-release durations.
    REMOVE candidates and every UNKNOWN frame are excluded from the reduction.
    """
    if (logits.shape != target.shape or logits.shape != mask.shape
            or logits.shape != teacher.shape or logits.ndim != 2 or logits.shape[1] != 40
            or note_target.shape != logits.shape[:1] or note_target.dtype != torch.int64
            or weights.shape != note_target.shape
            or any(v.requires_grad for v in (target, mask, note_target, teacher, weights))
            or not all(torch.isfinite(v).all() for v in (logits, target, mask, teacher, weights))
            or torch.any((target < 0) | (target > 1)) or torch.any((mask != 0) & (mask != 1))
            or torch.any((note_target != 0) & (note_target != 1)) or torch.any(weights <= 0)):
        raise ValueError("Invalid known-positive support anchor")
    known = (target[:, 8:32] > 0) & (mask[:, 8:32] == 1)
    selected = (note_target == 0) & known.any(dim=1)
    if not selected.any():
        return logits.reshape(-1)[:0].sum()
    allowed = known[selected]
    current = logits[selected, 8:32].masked_fill(~allowed, -torch.inf).amax(dim=1)
    original = teacher[selected, 8:32].masked_fill(~allowed, -torch.inf).amax(dim=1)
    floor = original.clamp(min=math.log(.25/.75))
    return ((floor-current).clamp_min(0).square()*weights[selected]).mean()


def freeze_support(network):
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(name.startswith("reject."))
    return {name: value.detach().clone() for name, value in network.state_dict().items()
            if not name.startswith("reject.")}


def require_frozen_support(network, original):
    current = {name: value for name, value in network.state_dict().items() if not name.startswith("reject.")}
    if current.keys() != original.keys() or any(not torch.equal(current[k], v) for k, v in original.items()):
        raise ValueError("Approved support branch changed during rejection-only fitting")


def teacher_outputs(network, inputs):
    network.eval()
    notes, frames = [], []
    with torch.inference_mode():
        for first in range(0, len(inputs[0]), 256):
            note, frame = network(*(v[first:first+256] for v in inputs))
            notes.append(note[:, 1]-note[:, 0])
            frames.append(frame)
    return torch.cat(notes).clone(), torch.cat(frames).clone()
