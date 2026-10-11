"""Hash-checked release corrections after approved accompaniment note filtering."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.attack_local_evidence import observe, sequences
from app.services.bass_harmonic import features
from app.services.chord_release_network import SHIFTS, VERSION, model, probabilities
from app.services.note_evidence import note_features
from app.services.release_local_evidence import release_sequences
from app.services.release_neighbors import features as neighbors


def correct(events, probability, duration, threshold):
    events, probability = np.asarray(events, float), np.asarray(probability)
    n = len(events)
    if (events.shape != (n, 4) or probability.shape != (n, 7)
            or not np.isfinite(events).all() or not np.isfinite(probability).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > duration) or not np.isfinite(duration) or duration <= 0
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-5)
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid release correction input")
    eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                & (events[:, 2] >= 21) & (events[:, 2] < 60))
    shift = np.where(eligible & (probability.max(axis=1) >= threshold),
                     SHIFTS[probability.argmax(axis=1)], 0.)
    result = events.copy()
    result[:, 1] += shift
    invalid = (result[:, 1]-result[:, 0] < .045) | (result[:, 1] > duration-2.5)
    result[invalid, 1] = events[invalid, 1]
    for pitch in np.unique(events[:, 2]):
        indices = np.flatnonzero(events[:, 2] == pitch)
        indices = indices[np.argsort(events[indices, 0], kind="stable")]
        for first, second in zip(indices, indices[1:], strict=False):
            touching = abs(events[first, 1]-events[second, 0]) < 1e-6
            collision = events[first, 1] <= events[second, 0] and result[first, 1] > events[second, 0]
            if ((touching and abs(result[first, 1]-events[second, 0]) > 1e-6)
                    or collision or events[first, 0] == events[second, 0]):
                result[[first, second], 1] = events[[first, second], 1]
    return result


class ReleaseModel:
    def __init__(self, saved):
        import torch

        metadata = {"version", "format_version", "width", "threshold", "mean", "scale"}
        if (not metadata.issubset(saved.files)
                or any(saved[k].shape != () for k in metadata-{"mean", "scale"})
                or str(saved["version"]) != VERSION or float(saved["format_version"]) != 4.
                or float(saved["width"]) not in (16., 24.)
                or float(saved["threshold"]) not in (.8, .9, .95, .99, .995, .999)):
            raise ValueError("Incompatible release specialist metadata")
        self.threshold = float(saved["threshold"])
        self.network = model(int(saved["width"]))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {"weight_"+key for key in template}:
            raise ValueError("Incompatible release specialist arrays")
        state = {}
        for key, tensor in template.items():
            array = saved["weight_"+key]
            if (array.dtype != np.float32 or array.shape != tuple(tensor.shape)
                    or not np.isfinite(array).all()):
                raise ValueError("Invalid release specialist weights")
            state[key] = torch.from_numpy(array.copy())
        self.normalizer = tuple(saved[key].copy() for key in ("mean", "scale"))
        if any(a.dtype != np.float32 or a.shape != (100,) or not np.isfinite(a).all()
               for a in self.normalizer) or np.any(self.normalizer[1] <= 0):
            raise ValueError("Invalid release specialist normalization")
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, item):
        return probabilities(self.network, item, self.normalizer)[2]


class ChordRelease:
    name = "Independent chord release specialist v1"

    def __init__(self, asset, expected_sha):
        asset = Path(asset)
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected_sha:
            raise PipelineError("The chord release model is missing or damaged. Rebuild the backend.")
        try:
            with np.load(asset, allow_pickle=False) as saved:
                self.model = ReleaseModel(saved)
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError("The chord release model contains incompatible data. Rebuild the backend.") from exc

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        samples, rate = sf.read(str(path), dtype="float32")
        if rate != 22050 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise PipelineError("Chord release correction needs finite mono 22050 Hz audio.")
        duration = len(samples)/rate
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        if not np.isfinite(events).all() or np.any(events[:, 1] > duration):
            return 0
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        if not eligible.any():
            return 0
        base = note_features(samples, rate, acoustic, notes, include_context=True)
        observed = observe(samples, rate)
        item = {"events": events, "x": features(events, base),
                "release_neighbors": neighbors(events, base),
                "attack_frames": sequences(observed, events),
                "release_frames": release_sequences(observed, events)}
        result = correct(events, self.model.probability(item), duration, self.model.threshold)
        changed = np.flatnonzero(result[:, 1] != events[:, 1])
        for index in changed:
            notes[index].end = float(result[index, 1])
        return len(changed)
