"""Verify accompaniment candidates with a checked, locally trained network."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.note_evidence import FEATURE_NAMES, FEATURE_VERSION, note_features

CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets' / 'accompaniment-verifier-v2.pt'
CHECKPOINT_SHA256 = '2b75ac2d3c80a2644c7df5c0ec5fab9c10086ff2fa072177868e7e6c3091fe60'


class AccompanimentVerifier:
    name = 'Accompaniment note verifier v2'

    def __init__(self):
        import torch

        from app.services.note_verifier import NoteVerifier

        if not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise PipelineError('The accompaniment model is missing or damaged. Rebuild the backend and retry.')
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if saved['feature_version'] != FEATURE_VERSION or saved['feature_names'] != list(FEATURE_NAMES):
            raise PipelineError('The accompaniment model and audio features have incompatible versions.')
        self.model = NoteVerifier().eval()
        self.model.load_state_dict(saved['state_dict'])
        self.mean, self.scale = saved['mean'].numpy(), saved['scale'].numpy()
        self.threshold = float(saved['threshold'])
        if (self.mean.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or not 0 <= self.threshold <= 1):
            raise PipelineError('The accompaniment model has invalid normalization or threshold data.')

    def filter(self, path, acoustic, midi):
        """Keep original pitch, timing and velocity; reject unsupported events only."""
        import torch

        notes = [n for part in midi.instruments for n in part.notes]
        if not notes:
            return 0
        samples, rate = sf.read(path, dtype='float32')
        if samples.ndim != 1 or rate != 22050 or not 0 < len(samples) <= 33 * rate:
            raise PipelineError('Accompaniment verification needs a normalized window of at most 33 seconds.')
        x = note_features(samples, rate, acoustic, notes)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(torch.from_numpy((x - self.mean) / self.scale))).numpy()
        if not np.isfinite(probability).all():
            raise PipelineError('The accompaniment model returned invalid note confidence.')
        keep = probability >= self.threshold
        index = 0
        for part in midi.instruments:
            size = len(part.notes)
            part.notes = [note for note, accepted in zip(part.notes, keep[index:index + size], strict=True) if accepted]
            index += size
        return int(np.sum(~keep))
