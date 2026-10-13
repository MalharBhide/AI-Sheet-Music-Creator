"""Condition frame support on raw attack and release evidence as well as CQT."""

from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v40 import model as periodic_model
from left_precision_v43 import gate, probabilities

VERSION = "boundary-conditioned-frame-support-v48-v1"
__all__ = ["VERSION", "model", "gate", "probabilities"]


def model(width):
    import torch
    from torch import nn

    class BoundaryPresence(nn.Module):
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
            self.condition = nn.Linear(CONTEXT_WIDTH+4*width, width)
            self.frame = nn.Conv1d(width, 1, 1)
            nn.init.zeros_(self.frame.weight)
            nn.init.constant_(self.frame.bias, 4.)

        def forward(self, coarse, attack, release, context):
            boundaries = []
            for encoder, window in ((self.attack_condition, attack), (self.release_condition, release)):
                encoded = encoder(window.transpose(1, 2))
                boundaries.extend((encoded.mean(dim=2), encoded.amax(dim=2)))
            condition = self.condition(torch.cat((context, *boundaries), dim=1)).unsqueeze(2)
            hidden = self.presence(coarse.transpose(1, 2))+condition
            return self.reject(coarse, attack, release, context), self.frame(torch.relu(hidden)).squeeze(1)

    return BoundaryPresence()
