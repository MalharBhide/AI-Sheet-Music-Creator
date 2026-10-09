"""Validated spectral/recurrence rejection after the unchanged V25 bass stage."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.attack_local_evidence import observe
from app.services.attack_local_evidence import sequences as attack_sequences
from app.services.bass_attack_declutter import AttackSupportModel
from app.services.bass_embedding_network import VERSION, model, probabilities
from app.services.bass_harmonic import BassHarmonic, features
from app.services.note_evidence import note_features
from app.services.release_local_evidence import release_sequences
from app.services.release_neighbors import features as neighbor_features
from app.services.temporal_note_model import sequences as coarse_sequences
from app.services.waveform_recurrence import features as recurrence_features


class EmbeddingSupportModel:
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
            or float(saved["format_version"]) != 3.0
            or float(saved["width"]) not in (64.0, 96.0)
            or float(saved["cap"]) not in (2.0, 4.0)
        ):
            raise ValueError("Incompatible spectral bass support checkpoint")
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
            raise ValueError("Invalid spectral bass confidence contract")
        self.network = model(int(saved["width"]), float(saved["cap"]))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {"weight_" + k for k in template}:
            raise ValueError("Incompatible spectral bass arrays")
        state = {}
        for key, tensor in template.items():
            array = saved["weight_" + key]
            if (
                array.dtype != np.float32
                or array.shape != tuple(tensor.shape)
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid spectral bass weights")
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
            raise ValueError("Invalid spectral bass normalization")
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, item):
        return probabilities(self.network, item, self.normalizer)


class BassEmbeddingDeclutter:
    name = "Bass spectral/recurrence verifier v1"

    def __init__(self, asset, expected_sha):
        asset = Path(asset)
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected_sha:
            raise PipelineError(
                "The spectral bass model is missing or damaged. Rebuild the backend and retry."
            )
        try:
            with np.load(asset, allow_pickle=False) as saved:
                self.model = EmbeddingSupportModel(saved)
            self.guardian = BassHarmonic().models[1]
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError(
                "The spectral bass model contains incompatible data. Rebuild the backend and retry."
            ) from exc

    def filter(self, path, acoustic, midi):
        notes = [n for p in midi.instruments for n in p.notes]
        if not notes:
            return 0
        samples, rate = sf.read(str(path), dtype="float32")
        if rate != 22050 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise PipelineError("Spectral bass verification needs finite mono 22050 Hz audio.")
        duration = len(samples) / rate
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        if np.any(events[:, 1] > duration) or not np.isfinite(events).all():
            return 0
        eligible = (
            (events[:, 0] >= 2.5)
            & (events[:, 1] <= duration - 2.5)
            & (events[:, 2] >= 21)
            & (events[:, 2] < 60)
        )
        if not np.any(eligible):
            return 0
        base = note_features(samples, rate, acoustic, notes, include_context=True)
        observed = observe(samples, rate)
        item = {
            "events": events,
            "x": features(events, base),
            "frames": coarse_sequences(observed["cqt"], events),
            "attack_frames": attack_sequences(observed, events),
            "release_frames": release_sequences(observed, events),
            "release_neighbors": neighbor_features(events, base),
            "periodicity": recurrence_features(samples, rate, events),
        }
        probability = self.model.probability(item)
        keep = self.model.keep(events, probability, self.guardian.probability(base), duration)
        accepted = {id(n) for n, k in zip(notes, keep, strict=True) if k}
        for part in midi.instruments:
            part.notes = [n for n in part.notes if id(n) in accepted]
        return int((~keep).sum())
