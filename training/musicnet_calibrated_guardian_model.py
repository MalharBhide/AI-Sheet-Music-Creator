"""Learn two acoustic decisions while preserving the complete released V32 anchor."""

import math

from musicnet_full_encoding_model import WARM_SHA, initialize, normalizer
from teacher_embedding_model import model as approved_model
from teacher_embedding_model import probabilities as checked_probabilities

VERSION = "musicnet-calibrated-protection-v35-v1"
__all__ = ["VERSION", "WARM_SHA", "initialize", "normalizer", "model", "probabilities"]


def model(width, cap):
    import torch
    from torch import nn

    if width not in (128, 192) or cap not in (4.0, 6.0):
        raise ValueError("Changed calibrated protection architecture")

    class CalibratedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = approved_model(64, 2.0)
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)
            self.correction = self.branch(width)
            self.protection = self.branch(width)
            self.cap = float(cap)
            with torch.no_grad():
                for branch in (self.correction, self.protection):
                    branch[-1].weight.zero_()
                    branch[-1].bias.zero_()
                # Start by protecting all events at every declared gate.
                self.protection[-1].bias.fill_(math.log(99.0))

        @staticmethod
        def branch(width):
            return nn.Sequential(
                nn.Linear(955, width),
                nn.ReLU(),
                nn.Linear(width, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def both(self, frames, attacks, endings, context, recurrence):
            self.anchor.eval()
            teacher = self.anchor.teacher
            coarse = teacher.coarse(frames.transpose(1, 2))
            attack = teacher.local(attacks.transpose(1, 2))
            release = teacher.ending(endings.transpose(1, 2))
            hidden = teacher.classifier[:-1](torch.cat((coarse, context, attack, release), dim=1))
            original = teacher.classifier[-1](hidden)
            old_delta = self.anchor.cap * torch.tanh(
                self.anchor.correction(torch.cat((recurrence, torch.log1p(hidden)), dim=1))
            )
            features = torch.cat(
                (
                    recurrence,
                    context,
                    torch.log1p(coarse),
                    torch.log1p(attack),
                    torch.log1p(release),
                    torch.log1p(hidden),
                ),
                dim=1,
            )
            delta = self.cap * torch.tanh(self.correction(features))
            logits = torch.cat((original[:, :1], original[:, 1:] + old_delta + delta), dim=1)
            return logits, self.protection(features).squeeze(1)

        def forward(self, *args):
            return self.both(*args)[0]

    return CalibratedNet()


def probabilities(network, item, normalization):
    """Return removal confidence and independently learned acoustic protection."""
    import torch
    from torch import nn

    class ProtectionView(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, *args):
            logit = self.source.both(*args)[1]
            return torch.stack((torch.zeros_like(logit), logit), dim=1)

    removal = checked_probabilities(network, item, normalization)
    protection = checked_probabilities(ProtectionView(network), item, normalization)[:, 1]
    return removal, protection
