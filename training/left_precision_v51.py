"""Warm rejection and fresh boundary frame support without frame distillation."""

from left_precision_v43 import model as released_model
from left_precision_v48 import gate, probabilities
from left_precision_v48 import model as boundary_model
from left_precision_v50 import real_piano_weights

VERSION = "warm-rejection-fresh-presence-v51-v1"
FRESH_SEEDS = tuple(range(296101, 296141))
__all__ = ["VERSION", "FRESH_SEEDS", "gate", "initialize", "model", "probabilities",
           "real_piano_weights", "rejection_rehearsal_loss"]


def model(width):
    if width != 12:
        raise ValueError("Warm rejection requires released width 12")
    return boundary_model(width)


def initialize(teacher):
    import torch

    expected = released_model(12).state_dict()
    source = teacher.state_dict()
    if set(source) != set(expected) or any(source[n].shape != expected[n].shape
            or source[n].dtype != torch.float32 or not torch.isfinite(source[n]).all() for n in expected):
        raise ValueError("Changed released rejection architecture or parameters")
    network = model(12)
    network.reject.load_state_dict({n.removeprefix("reject."): v.detach().clone()
        for n, v in source.items() if n.startswith("reject.")}, strict=True)
    return network


def rejection_rehearsal_loss(logits, teacher):
    """Fitting-only soft rejection regularizer; frame predictions are not targets."""
    import torch

    if (logits.ndim != 2 or logits.shape[1] != 2 or teacher.shape != logits.shape
            or not torch.isfinite(logits).all() or not torch.isfinite(teacher).all()
            or torch.any(teacher < 0) or torch.any(teacher > 1)
            or not torch.allclose(teacher.sum(dim=1), torch.ones_like(teacher[:, 0]), atol=1e-5)):
        raise ValueError("Invalid fitting rejection rehearsal")
    return torch.nn.functional.kl_div(torch.log_softmax(logits, dim=1),
        teacher.detach(), reduction="none").sum(dim=1)
