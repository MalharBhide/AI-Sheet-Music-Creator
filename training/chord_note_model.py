"""Experimental piano note verifier with simultaneous-note context.

This is an offline candidate, not the website's released transcription model.
It reuses the released note encoder; only the contextual residual is trained.
No chord dictionary, reference pitches, or fabricated notes enter inference.
"""

import numpy as np
import torch
from app.services.note_verifier import NoteVerifier
from torch import nn


def peer_matrix(events, tolerance=.04):
    """Same-attack peers, excluding self and duplicate detections of one key.

    Pairwise distance avoids chaining an arpeggio into one simultaneous chord.
    The matrix is independent of event ordering and contains no label data.
    """
    events = np.asarray(events, dtype=float).reshape(-1, 4)
    return ((np.abs(events[:, 0, None] - events[None, :, 0]) <= tolerance)
            & (events[:, 2, None] != events[None, :, 2]))


class ChordNoteVerifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.base = NoteVerifier()
        self.base.load_state_dict(checkpoint['state_dict'])
        self.base.requires_grad_(False)
        self.register_buffer('mean', checkpoint['mean'].clone())
        self.register_buffer('scale', checkpoint['scale'].clone())
        # Own embedding, mean/max peer embeddings, relative pitch classes,
        # peer count and the original note logit.
        self.residual = nn.Sequential(nn.Linear(50, 24), nn.ReLU(), nn.Linear(24, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, events):
        x = (features - self.mean) / self.scale
        embedded = self.base.network[:4](x)
        original = self.base.network[4](embedded).squeeze(-1)
        if not len(features):
            return original
        peers = torch.as_tensor(peer_matrix(events), device=x.device)
        count = peers.sum(dim=1, keepdim=True)
        mean = peers.to(x.dtype) @ embedded / count.clamp_min(1)
        maximum = embedded[None, :, :].expand(len(x), -1, -1).masked_fill(
            ~peers[:, :, None], -torch.inf).amax(dim=1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        pitches = torch.as_tensor(np.asarray(events)[:, 2], device=x.device).long()
        intervals = (pitches[None, :] - pitches[:, None]) % 12
        chroma = torch.stack([(peers & (intervals == k)).sum(dim=1) for k in range(12)], dim=1)
        chroma = chroma.to(x.dtype) / count.clamp_min(1)
        context = torch.cat((embedded, mean, maximum, chroma,
                             count.to(x.dtype).log1p(), original[:, None]), dim=1)
        return original + self.residual(context).squeeze(-1)
