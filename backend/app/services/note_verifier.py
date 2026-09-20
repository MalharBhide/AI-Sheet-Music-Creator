"""Small supervised classifier for whether an acoustic note is supported."""

from torch import nn

from app.services.note_evidence import FEATURE_NAMES


class NoteVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 24), nn.ReLU(),
                                     nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)
