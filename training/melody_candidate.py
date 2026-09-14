"""Research candidate: warm-started melody decoder with longer residual context."""

import torch
from torch import nn

from app.services.melody_decoder import CHANNELS, MelodyDecoder


class ContextMelodyDecoder(MelodyDecoder):
    """Retain V1 predictions initially, then learn a 1.26 second correction."""

    def __init__(self):
        super().__init__()
        layers = [nn.Conv1d(CHANNELS, 16, 3, padding=1), nn.ReLU()]
        for dilation in (2, 4, 8, 16):
            layers.extend([nn.Conv1d(16, 16, 3, padding=dilation, dilation=dilation), nn.ReLU()])
        layers.append(nn.Conv1d(16, 2, 1))
        self.context = nn.Sequential(*layers)
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, x):
        logits, attacks = super().forward(x)
        batch, time, pitch, channels = x.shape
        shared = x.permute(0, 2, 3, 1).reshape(batch * pitch, channels, time)
        result = self.context(shared).reshape(batch, pitch, 2, time)
        correction = result[:, :, 0].transpose(1, 2)
        return torch.cat([logits[:, :, :88] + correction, logits[:, :, 88:]], dim=-1), (
            attacks + result[:, :, 1].transpose(1, 2))
