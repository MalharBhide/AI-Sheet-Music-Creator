"""Learn from all frozen acoustic time positions while retaining complete V32."""

import numpy as np

from app.services.bass_embedding_network import model as approved_model

VERSION = "musicnet-full-encodings-v34-v1"
WARM_SHA = "78fcb4f2c0effdac1dfbb7de54efba88a63429aeb358c9c02620c360764122eb"


def model(width, cap):
    import torch
    from torch import nn

    if width not in (128, 192) or cap not in (4.0, 6.0):
        raise ValueError("Changed MusicNet anchored architecture")

    class AnchoredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = approved_model(64, 2.0)
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)
            self.correction = nn.Sequential(
                nn.Linear(955, width),
                nn.ReLU(),
                nn.Linear(width, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.cap = float(cap)
            with torch.no_grad():
                self.correction[-1].weight.zero_()
                self.correction[-1].bias.zero_()

        def forward(self, frames, attacks, endings, context, recurrence):
            self.anchor.eval()
            teacher = self.anchor.teacher
            coarse = teacher.coarse(frames.transpose(1, 2))
            local = teacher.local(attacks.transpose(1, 2))
            ending = teacher.ending(endings.transpose(1, 2))
            acoustic = torch.cat((coarse, context, local, ending), dim=1)
            hidden = teacher.classifier[:-1](acoustic)
            logits = teacher.classifier[-1](hidden)
            inputs = torch.cat((recurrence, torch.log1p(hidden)), dim=1)
            original_delta = self.anchor.cap * torch.tanh(self.anchor.correction(inputs))
            # Retain the eight pooled time positions of all frozen encoders,
            # rather than discarding them at the teacher's 32-value bottleneck.
            detailed = torch.cat(
                (
                    recurrence,
                    context,
                    torch.log1p(coarse),
                    torch.log1p(local),
                    torch.log1p(ending),
                    torch.log1p(hidden),
                ),
                dim=1,
            )
            new_delta = self.cap * torch.tanh(self.correction(detailed))
            return torch.cat((logits[:, :1], logits[:, 1:] + original_delta + new_delta), dim=1)

    return AnchoredNet()


def initialize(network, saved):
    from app.services.bass_embedding_network import VERSION as approved_version

    if (
        saved["version"] != approved_version
        or saved["width"] != 64
        or saved["cap"] != 2.0
        or saved["mean"].shape != (155,)
        or saved["scale"].shape != (155,)
    ):
        raise ValueError("Changed approved V32 anchor checkpoint or normalizer")
    network.anchor.load_state_dict(saved["state_dict"], strict=True)
    network.anchor.eval()
    return network


def probabilities(network, item, normalizer):
    # The sealed V32 physical input validator and normalization remain exact.
    # Its helper accepts any network with this five-tensor forward interface.
    from app.services.bass_embedding_network import probabilities as approved_probabilities

    return approved_probabilities(network, item, normalizer)


def normalizer(saved):
    mean, scale = (saved[k].numpy().copy() for k in ("mean", "scale"))
    if (
        mean.dtype != np.float32
        or scale.dtype != np.float32
        or mean.shape != (155,)
        or scale.shape != (155,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid frozen V32 normalizer")
    return mean, scale
