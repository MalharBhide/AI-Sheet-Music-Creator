"""Calibrate the existing final support layer without changing learned features."""

import torch
from left_precision_v45 import masked_frame_loss

MUTABLE = frozenset(("frame.weight", "frame.bias"))


def freeze_backbone(network):
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(name in MUTABLE)
    if sum(p.numel() for p in network.parameters() if p.requires_grad) != 13:
        raise ValueError("Only the approved 12-channel support readout may change")
    return {name: value.detach().clone() for name, value in network.state_dict().items() if name not in MUTABLE}


def require_frozen_backbone(network, original):
    current = {name: value for name, value in network.state_dict().items() if name not in MUTABLE}
    if current.keys() != original.keys() or any(not torch.equal(current[k], v) for k, v in original.items()):
        raise ValueError("Approved feature extractor or rejection classifier changed")


def keep_rehearsal(logits, teacher, mask, target, weights):
    if (logits.ndim != 2 or logits.shape[1] != 40 or teacher.shape != logits.shape or mask.shape != logits.shape
            or target.shape != logits.shape[:1] or target.dtype != torch.int64 or weights.shape != target.shape
            or any(v.requires_grad for v in (teacher, mask, target, weights))
            or not all(torch.isfinite(v).all() for v in (logits, teacher, mask, weights))
            or torch.any((mask != 0) & (mask != 1)) or torch.any((target != 0) & (target != 1))
            or torch.any(weights <= 0)):
        raise ValueError("Invalid fitting-only KEEP rehearsal")
    selected = (target == 0) & mask.any(dim=1)
    if not selected.any():
        return logits.reshape(-1)[:0].sum()
    probability = torch.sigmoid(teacher[selected])
    safe = probability.clamp(1e-7, 1-1e-7)
    entropy = -(probability*safe.log()+(1-probability)*(1-safe).log())
    loss = masked_frame_loss(logits[selected], probability, mask[selected], None)
    loss -= (entropy*mask[selected]).sum(dim=1)/mask[selected].sum(dim=1).clamp_min(1)
    return (loss*weights[selected]).mean()
