"""A separately trained endpoint specialist; pitch and attacks stay immutable."""

import numpy as np
from chord_boundary_v36 import STAY
from chord_boundary_v36 import probabilities as boundary_probabilities
from protected_boundary_targets_v37 import protected_labels

VERSION = "protected-release-only-v38-v1"


def labels(item):
    result = protected_labels(item)
    result[:, :2] = -100
    # Unmatched and unresolvable eligible tails must teach abstention too.
    result[item["eligible"] & (result[:, 2] == -100), 2] = STAY
    return result


def model(width):
    import torch
    from torch import nn

    if width not in (16, 24):
        raise ValueError("Invalid release-only encoder")

    class ReleaseOnly(nn.Module):
        def __init__(self):
            super().__init__()

            def encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(16), nn.Flatten())

            self.attack, self.release = encoder(), encoder()
            self.ending = nn.Sequential(nn.Linear(2*width*16+100, 96), nn.ReLU(),
                                        nn.Dropout(.1), nn.Linear(96, 7))
            nn.init.zeros_(self.ending[-1].weight)
            nn.init.zeros_(self.ending[-1].bias)
            with torch.no_grad():
                self.ending[-1].bias[STAY] = 2

        def forward(self, attacks, releases, observed):
            hidden = torch.cat((self.attack(attacks.transpose(1, 2)),
                                self.release(releases.transpose(1, 2)), observed), dim=1)
            # Deterministic inert heads let the unchanged musical evaluator check
            # that neither note support nor attacks can be changed by this model.
            keep = hidden.new_full((len(hidden), 2), -100.)
            attack = hidden.new_full((len(hidden), 7), -100.)
            keep[:, 0], attack[:, STAY] = 0., 0.
            return keep, attack, self.ending(hidden)

    return ReleaseOnly()


def probabilities(network, item, normalizer):
    result = boundary_probabilities(network, item, normalizer)
    if (np.any(result[0][:, 1] > 1e-10)
            or np.any(result[1].argmax(axis=1) != STAY)):
        raise ValueError("Release specialist changed its immutable heads")
    return result
