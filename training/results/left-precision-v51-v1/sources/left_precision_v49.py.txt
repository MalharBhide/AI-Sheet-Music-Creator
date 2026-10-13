"""Warm-started boundary support with an exactly preserved V43 physical path."""

from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v40 import model as periodic_model
from left_precision_v43 import gate, probabilities

VERSION = "warm-boundary-support-v49-v1"
__all__ = ["VERSION", "model", "initialize", "gate", "probabilities"]


def model(width):
    import torch
    from torch import nn

    if width != 12:
        raise ValueError("Warm boundary support requires released width 12")

    class WarmBoundaryPresence(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = periodic_model(width)
            self.presence = nn.Sequential(nn.Conv1d(9, width, 5, padding=2), nn.ReLU(),
                nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            def boundary_encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            self.attack_condition = boundary_encoder()
            self.release_condition = boundary_encoder()
            # Keep the physical projection's shape and arithmetic unchanged.
            self.condition = nn.Linear(CONTEXT_WIDTH, width)
            self.boundary_condition = nn.Linear(4*width, width, bias=False)
            nn.init.zeros_(self.boundary_condition.weight)
            self.frame = nn.Conv1d(width, 1, 1)
            nn.init.zeros_(self.frame.weight)
            nn.init.constant_(self.frame.bias, 4.)

        def forward(self, coarse, attack, release, context):
            boundaries = []
            for encoder, window in ((self.attack_condition, attack), (self.release_condition, release)):
                encoded = encoder(window.transpose(1, 2))
                boundaries.extend((encoded.mean(dim=2), encoded.amax(dim=2)))
            hidden = self.presence(coarse.transpose(1, 2))+self.condition(context).unsqueeze(2)
            hidden = hidden+self.boundary_condition(torch.cat(boundaries, dim=1)).unsqueeze(2)
            return self.reject(coarse, attack, release, context), self.frame(torch.relu(hidden)).squeeze(1)

    return WarmBoundaryPresence()


def initialize(teacher):
    import torch

    network = model(12)
    source, target = teacher.state_dict(), network.state_dict()
    original = {name for name in target if not name.startswith(
        ("attack_condition.", "release_condition.", "boundary_condition."))}
    if set(source) != original or any(source[n].shape != target[n].shape
            or source[n].dtype != torch.float32 or not torch.isfinite(source[n]).all() for n in original):
        raise ValueError("Changed released frame-support architecture or parameters")
    for name in original:
        target[name] = source[name].detach().clone()
    network.load_state_dict(target, strict=True)
    return network
