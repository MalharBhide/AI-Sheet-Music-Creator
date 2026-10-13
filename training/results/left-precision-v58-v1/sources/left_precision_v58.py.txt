"""Fresh frame-support learner with explicit candidate-relative frame phases."""

from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v40 import model as periodic_model
from left_precision_v43 import gate, probabilities
from left_precision_v57 import (
    frame_supervision,
    hard_keep_loss,
    isolated_negative_weights,
    real_piano_weights,
    validate_partitions,
)

VERSION = "candidate-phase-conditioned-presence-v58-v1"
FRESH_SEEDS = tuple(range(302101, 302141))
__all__ = ["VERSION", "FRESH_SEEDS", "model", "gate", "probabilities", "frame_phases",
           "require_phase_contract", "frame_supervision", "hard_keep_loss", "isolated_negative_weights",
           "real_piano_weights", "validate_partitions"]


def frame_phases():
    """Match the existing 8-before/24-inside/8-after observation clock.

    The coordinate is frame rank, not elapsed seconds. Interior samples span a
    variable candidate duration; the exterior samples span fixed local windows.
    No label, source identity, absolute onset or tempo enters these channels.
    """
    import torch

    phase = torch.zeros(1, 4, 40, dtype=torch.float32)
    phase[0, 0] = torch.linspace(-1., 1., 40)
    phase[0, 1, :8] = 1
    phase[0, 2, 8:32] = 1
    phase[0, 3, 32:] = 1
    return phase


def require_phase_contract(network):
    import torch

    expected = frame_phases().to(network.phase.device)
    if (network.phase.dtype != torch.float32 or network.phase.requires_grad
            or not torch.equal(network.phase, expected)):
        raise ValueError("Changed candidate-relative frame phase clock")


def model(width):
    import torch
    from torch import nn

    class PhasePresence(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = periodic_model(width)
            self.register_buffer("phase", frame_phases())
            self.presence = nn.Sequential(nn.Conv1d(13, width, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            def boundary_encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            self.attack_condition = boundary_encoder()
            self.release_condition = boundary_encoder()
            self.condition = nn.Linear(CONTEXT_WIDTH+4*width, width)
            self.frame = nn.Conv1d(width, 1, 1)
            nn.init.zeros_(self.frame.weight)
            nn.init.constant_(self.frame.bias, 4.)

        def forward(self, coarse, attack, release, context):
            count = len(context)
            if (coarse.shape != (count, 40, 9) or attack.shape != (count, 61, 18)
                    or release.shape != (count, 61, 18) or context.shape != (count, CONTEXT_WIDTH)):
                raise ValueError("Changed physical candidate observation clock")
            boundaries = []
            for encoder, window in ((self.attack_condition, attack), (self.release_condition, release)):
                encoded = encoder(window.transpose(1, 2))
                boundaries.extend((encoded.mean(dim=2), encoded.amax(dim=2)))
            condition = self.condition(torch.cat((context, *boundaries), dim=1)).unsqueeze(2)
            phased = torch.cat((coarse.transpose(1, 2), self.phase.expand(count, -1, -1)), dim=1)
            hidden = self.presence(phased)+condition
            return self.reject(coarse, attack, release, context), self.frame(torch.relu(hidden)).squeeze(1)

    return PhasePresence()
