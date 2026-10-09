"""Load a gated MusicNet correction; this module is not wired into the website."""

import hashlib
from pathlib import Path

import numpy as np

from app.models import PipelineError
from app.services.bass_attack_declutter import AttackSupportModel
from app.services.bass_embedding_declutter import BassEmbeddingDeclutter
from app.services.bass_harmonic import BassHarmonic
from app.services.bass_musicnet_network import VERSION, model, probabilities


class MusicNetSupportModel:
    keep = AttackSupportModel.keep

    def __init__(self, saved):
        import torch

        metadata = {
            "version",
            "format_version",
            "width",
            "cap",
            "remove_threshold",
            "guardian_threshold",
            "mean",
            "scale",
        }
        if (
            not metadata.issubset(saved.files)
            or any(saved[k].shape != () for k in metadata - {"mean", "scale"})
            or str(saved["version"]) != VERSION
            or float(saved["format_version"]) != 4.0
            or float(saved["width"]) not in (64.0, 96.0)
            or float(saved["cap"]) not in (2.0, 4.0)
        ):
            raise ValueError("Incompatible MusicNet bass support checkpoint")
        self.remove_threshold, self.guardian_threshold = (
            float(saved["remove_threshold"]),
            float(saved["guardian_threshold"]),
        )
        if self.remove_threshold not in (
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ) or self.guardian_threshold not in (0.15, 0.3, 0.5):
            raise ValueError("Invalid MusicNet bass confidence contract")
        self.network = model(int(saved["width"]), float(saved["cap"]))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {"weight_" + k for k in template}:
            raise ValueError("Incompatible MusicNet bass arrays")
        state = {}
        for key, tensor in template.items():
            array = saved["weight_" + key]
            if (
                array.dtype != np.float32
                or array.shape != tuple(tensor.shape)
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid MusicNet bass weights")
            state[key] = torch.from_numpy(array.copy())
        self.normalizer = tuple(saved[k].copy() for k in ("mean", "scale"))
        mean, scale = self.normalizer
        if (
            mean.dtype != np.float32
            or scale.dtype != np.float32
            or mean.shape != (155,)
            or scale.shape != (155,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Invalid MusicNet bass normalization")
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, item):
        return probabilities(self.network, item, self.normalizer)


class BassMusicNetDeclutter(BassEmbeddingDeclutter):
    # Use the exact released physical feature extraction and unchanged-event mask.
    name = "Bass real-piano verifier v1"

    def __init__(self, asset, expected_sha):
        asset = Path(asset)
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected_sha:
            raise PipelineError("The real-piano bass model is missing or damaged.")
        try:
            with np.load(asset, allow_pickle=False) as saved:
                self.model = MusicNetSupportModel(saved)
            self.guardian = BassHarmonic().models[1]
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError("The real-piano bass model contains incompatible data.") from exc
