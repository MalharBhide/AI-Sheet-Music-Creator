"""Prospective NPZ runtime for V58; never routed to production before release gates.

This implementation uses the deployed physical-feature reader and independent
runtime network construction. It does not import the fitting model definition.
Its additional permissions must be intersected with immutable V43 permissions.
"""

import numpy as np

VERSION = "candidate-phase-conditioned-presence-v58-v1"
FORMAT = 58


def runtime_model(width):
    import torch
    from app.services.left_presence_network import CONTEXT_WIDTH
    from app.services.left_presence_network import model as baseline_model
    from torch import nn

    class PhaseRuntime(nn.Module):
        def __init__(self):
            super().__init__()
            self.reject = baseline_model(width).reject
            phase = torch.zeros(1, 4, 40, dtype=torch.float32)
            phase[0, 0] = torch.linspace(-1., 1., 40)
            phase[0, 1, :8], phase[0, 2, 8:32], phase[0, 3, 32:] = 1, 1, 1
            self.register_buffer("phase", phase)
            self.presence = nn.Sequential(nn.Conv1d(13, width, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            def boundary_encoder():
                return nn.Sequential(nn.Conv1d(18, width, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(width, width, 5, padding=2), nn.ReLU())

            self.attack_condition = boundary_encoder()
            self.release_condition = boundary_encoder()
            self.condition = nn.Linear(CONTEXT_WIDTH+4*width, width)
            self.frame = nn.Conv1d(width, 1, 1)

        def forward(self, coarse, attack, release, context):
            count = len(context)
            if (coarse.shape != (count, 40, 9) or attack.shape != (count, 61, 18)
                    or release.shape != (count, 61, 18) or context.shape != (count, CONTEXT_WIDTH)):
                raise ValueError("Changed runtime candidate observation clock")
            boundaries = []
            for encoder, window in ((self.attack_condition, attack), (self.release_condition, release)):
                encoded = encoder(window.transpose(1, 2))
                boundaries.extend((encoded.mean(dim=2), encoded.amax(dim=2)))
            condition = self.condition(torch.cat((context, *boundaries), dim=1)).unsqueeze(2)
            phased = torch.cat((coarse.transpose(1, 2), self.phase.expand(count, -1, -1)), dim=1)
            return self.reject(coarse, attack, release, context), self.frame(
                torch.relu(self.presence(phased)+condition)).squeeze(1)

    return PhaseRuntime()


class AdditionalPhaseModel:
    def __init__(self, saved):
        import torch

        metadata = {"version", "format_version", "width", "threshold", "veto_threshold", "mean", "scale"}
        if (not metadata.issubset(saved.files)
                or any(saved[key].shape != () for key in metadata-{"mean", "scale"})
                or str(saved["version"]) != VERSION or float(saved["format_version"]) != FORMAT
                or float(saved["width"]) not in (12, 20)
                or float(saved["threshold"]) not in (.9, .95, .98, .99, .995, .999)
                or float(saved["veto_threshold"]) not in (.05, .1, .2, .3)):
            raise ValueError("Incompatible additional phase-model metadata")
        self.threshold = float(saved["threshold"])
        self.veto_threshold = float(saved["veto_threshold"])
        self.network = runtime_model(int(saved["width"]))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {"weight_"+key for key in template}:
            raise ValueError("Incompatible additional phase-model arrays")
        state = {}
        for key, tensor in template.items():
            array = saved["weight_"+key]
            if array.dtype != np.float32 or array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError("Invalid additional phase-model weights")
            state[key] = torch.from_numpy(array.copy())
        if not torch.equal(state["phase"], template["phase"]):
            raise ValueError("Changed fixed runtime phase clock")
        self.normalizer = tuple(saved[key].copy() for key in ("mean", "scale"))
        if (any(array.dtype != np.float32 or array.shape != (163,) or not np.isfinite(array).all()
                for array in self.normalizer) or np.any(self.normalizer[1] < np.float32(.01))):
            raise ValueError("Invalid additional phase-model normalization")
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, item, stricter=False):
        from app.services.left_presence_network import gated, heads

        return gated(heads(self.network, item, self.normalizer), self.veto_threshold*(.8 if stricter else 1))

    def keep(self, item, duration, stricter=False):
        from app.services.left_presence_network import keep

        threshold = min(.9999, self.threshold+.005) if stricter else self.threshold
        return keep(item["events"], self.probability(item, stricter), duration, threshold)

    def intersect(self, baseline_permissions, item, duration, stricter=False):
        baseline_permissions = np.asarray(baseline_permissions)
        if baseline_permissions.dtype != np.bool_ or baseline_permissions.shape != (len(item["events"]),):
            raise ValueError("Complete immutable baseline permissions required")
        return baseline_permissions & self.keep(item, duration, stricter)
