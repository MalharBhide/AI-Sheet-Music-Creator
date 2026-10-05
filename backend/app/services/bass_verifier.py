"""Conservative, trained note verification for the balanced bounded bass stem."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import CONTEXT_NAMES, CONTEXT_VERSION, FEATURE_NAMES, note_features

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
CHECKPOINTS = (
    (ASSETS / 'bass-positive-consensus-v2.npz',
     '9864e9c040629f5412fab624dbf565ebc04677d0d046ebf22d188da80dec00c7', FEATURE_NAMES + CONTEXT_NAMES),
    (ASSETS / 'bass-positive-guardian-v2.npz',
     '59e3b1cf6c2b27198c9aa046127f98a1aff15ee287a34e981113382c01f371dd', FEATURE_NAMES),
)
EDGE_SECONDS = 2.5


class BassVerifier:
    name = 'Bass note verifier v1 (trained conservative consensus)'

    def __init__(self):
        self.models = []
        for path, expected, names in CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass note model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as saved:
                    model = ContextNoteModel(saved, feature_names=names, feature_version=CONTEXT_VERSION)
                if model.threshold != .1:
                    raise ValueError('Unexpected bass consensus threshold')
            except (KeyError, ValueError, OSError) as exc:
                raise PipelineError('The bass note model has incompatible or invalid data. Rebuild the backend and retry.') from exc
            self.models.append(model)

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        samples, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise PipelineError('Bass verification needs finite mono audio at 22050 Hz.')
        x = note_features(samples, rate, acoustic, notes, include_context=True)
        context, guardian = [model.probability(x[:, :model.feature_count]) for model in self.models]
        duration = len(samples) / rate
        eligible = np.asarray([note.start >= EDGE_SECONDS and note.end <= duration - EDGE_SECONDS
                               and 21 <= note.pitch < 60 for note in notes], dtype=bool)
        keep = ~(eligible & (context < self.models[0].threshold) & (guardian < self.models[1].threshold))
        cursor = 0
        for part in midi.instruments:
            count = len(part.notes)
            part.notes = [note for note, accepted in zip(part.notes, keep[cursor:cursor + count], strict=True) if accepted]
            cursor += count
        return int((~keep).sum())
