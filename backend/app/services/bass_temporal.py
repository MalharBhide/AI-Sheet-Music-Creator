"""Temporal acoustic KEEP evidence for actual post-V14 bass candidates."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.bass_harmonic import CHECKPOINTS, features
from app.services.bass_harmonic import VERSION as GUARDIAN_VERSION
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import note_features
from app.services.temporal_note_model import (
    FEATURE_COUNT,
    VERSION,
    create_model,
    probability,
    sequences,
    spectrum,
)

ASSET = Path(__file__).resolve().parents[1] / 'assets/bass-temporal-v1.npz'
EXPECTED_SHA = '0edc221998bbaacb052fbffe43f6efcce6e759290d0cdb9a7314028521e162c4'


class TemporalCandidateModel:
    def __init__(self, saved):
        import torch

        if (str(saved['version']) != VERSION or int(saved['format_version']) != 1
                or int(saved['width']) != 48 or float(saved['threshold']) != .005
                or float(saved['guardian_threshold']) != .3):
            raise ValueError('Incompatible temporal checkpoint contract')
        self.threshold = .005
        self.guardian_threshold = .3
        self.model = create_model(48)
        template = self.model.state_dict()
        metadata = {'version', 'format_version', 'width', 'threshold', 'guardian_threshold', 'mean', 'scale'}
        if set(saved.files) != metadata | {'weight_' + key for key in template}:
            raise ValueError('Incompatible temporal checkpoint arrays')
        state = {}
        for key, tensor in template.items():
            array = saved['weight_' + key]
            if array.dtype != np.float32 or array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError('Invalid temporal checkpoint weights')
            state[key] = torch.from_numpy(array.copy())
        self.normalizer = tuple(np.asarray(saved[key], np.float32).copy() for key in ('mean', 'scale'))
        mean, scale = self.normalizer
        if (mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,)
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('Invalid temporal checkpoint normalization')
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def probability(self, frames, context):
        return probability(self.model, frames, context, self.normalizer)


class BassTemporal:
    name = 'Bass temporal CNN v1 (trained spectral note evidence)'

    def __init__(self):
        guardian_path, guardian_sha, names, threshold = CHECKPOINTS[1]
        for path, expected in ((ASSET, EXPECTED_SHA), (guardian_path, guardian_sha)):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass temporal model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(ASSET, allow_pickle=False) as saved:
                self.model = TemporalCandidateModel(saved)
            with np.load(guardian_path, allow_pickle=False) as saved:
                self.guardian = ContextNoteModel(saved, feature_names=names, feature_version=GUARDIAN_VERSION)
            if self.guardian.threshold != threshold or threshold != self.model.guardian_threshold:
                raise ValueError('Changed temporal guardian threshold')
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError('The bass temporal model contains incompatible data. Rebuild the backend and retry.') from exc

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Bass temporal verification needs finite mono audio at 22050 Hz.')
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        duration = len(waveform) / rate
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        if not np.any(eligible):
            return 0
        base_x = note_features(waveform, rate, acoustic, notes, include_context=True)
        x = features(events, base_x)
        frames = sequences(spectrum(waveform, rate), events)
        p, g = self.model.probability(frames, x), self.guardian.probability(base_x)
        keep = ~(eligible & (p < self.model.threshold) & (g < self.guardian.threshold))
        accepted = {id(n) for n, k in zip(notes, keep, strict=True) if k}
        for part in midi.instruments:
            part.notes = [n for n in part.notes if id(n) in accepted]
        return int((~keep).sum())
