"""Removal decisions before release correction, applied only after its full input."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.attack_local_evidence import observe, sequences
from app.services.bass_harmonic import features
from app.services.left_presence_network import CONTEXT_WIDTH, VERSION, gated, heads, keep, model
from app.services.note_evidence import note_features
from app.services.release_local_evidence import release_sequences
from app.services.release_neighbors import features as neighbors
from app.services.temporal_note_model import sequences as coarse_sequences
from app.services.waveform_recurrence import features as recurrence


class PresenceModel:
    def __init__(self, saved):
        import torch

        metadata = {"version", "format_version", "width", "threshold", "veto_threshold", "mean", "scale"}
        if (not metadata.issubset(saved.files)
                or any(saved[k].shape != () for k in metadata-{"mean", "scale"})
                or str(saved["version"]) != VERSION or float(saved["format_version"]) != 5
                or float(saved["width"]) != 12 or float(saved["threshold"]) != .9
                or float(saved["veto_threshold"]) != .2):
            raise ValueError("Incompatible pitch-presence checkpoint metadata")
        self.threshold, self.veto_threshold = float(saved["threshold"]), float(saved["veto_threshold"])
        self.network = model(int(saved["width"]))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {"weight_"+key for key in template}:
            raise ValueError("Incompatible pitch-presence arrays")
        state = {}
        for key, tensor in template.items():
            array = saved["weight_"+key]
            if array.dtype != np.float32 or array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError("Invalid pitch-presence weights")
            state[key] = torch.from_numpy(array.copy())
        self.normalizer = tuple(saved[k].copy() for k in ("mean", "scale"))
        if (any(a.dtype != np.float32 or a.shape != (CONTEXT_WIDTH,) or not np.isfinite(a).all()
                for a in self.normalizer) or np.any(self.normalizer[1] <= 0)):
            raise ValueError("Invalid pitch-presence normalizer")
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, item, stricter=False):
        return gated(heads(self.network, item, self.normalizer), self.veto_threshold*(.8 if stricter else 1))

    def keep(self, item, duration, stricter=False):
        return keep(item["events"], self.probability(item, stricter), duration,
                    min(.9999, self.threshold+.005) if stricter else self.threshold)


@dataclass(frozen=True)
class RemovalProposal:
    original: tuple[int, ...]
    accepted: frozenset[int]

    def apply(self, midi):
        current = tuple(id(n) for p in midi.instruments for n in p.notes)
        if current != self.original:
            raise PipelineError("Pitch-presence decisions no longer match the release input.")
        for part in midi.instruments:
            part.notes = [note for note in part.notes if id(note) in self.accepted]
        return len(current)-len(self.accepted)


class LeftHandPresence:
    name = "Left-hand pitch-presence verifier v1"

    def __init__(self, asset, expected_sha):
        asset = Path(asset)
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected_sha:
            raise PipelineError("The left-hand presence model is missing or damaged. Rebuild the backend.")
        try:
            with np.load(asset, allow_pickle=False) as saved:
                self.model = PresenceModel(saved)
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError("The left-hand presence model contains incompatible data. Rebuild the backend.") from exc

    def prepare(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        original = tuple(map(id, notes))
        unchanged = RemovalProposal(original, frozenset(original))
        if not notes:
            return unchanged
        samples, rate = sf.read(str(path), dtype="float32")
        if rate != 22050 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise PipelineError("Pitch-presence verification needs finite mono 22050 Hz audio.")
        duration = len(samples)/rate
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        if not np.isfinite(events).all() or np.any(events[:, 1] > duration):
            return unchanged
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        if not eligible.any():
            return unchanged
        base = note_features(samples, rate, acoustic, notes, include_context=True)
        observed = observe(samples, rate)
        item = {"events": events, "x": features(events, base), "release_neighbors": neighbors(events, base),
                "frames": coarse_sequences(observed["cqt"], events),
                "attack_frames": sequences(observed, events), "release_frames": release_sequences(observed, events),
                "periodicity": recurrence(samples, rate, events)}
        accepted = self.model.keep(item, duration)
        return RemovalProposal(original, frozenset(key for key, allowed in zip(original, accepted, strict=True) if allowed))
