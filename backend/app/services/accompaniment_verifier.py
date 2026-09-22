"""Verify accompaniment candidates with a checked, locally trained network."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.accompaniment_candidates import decode_candidates
from app.services.note_context_model import (
    ContextNoteModel,
    correction_keep,
    residual_keep,
    shared_events,
)
from app.services.note_evidence import FEATURE_NAMES, FEATURE_VERSION, note_features

CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets' / 'accompaniment-verifier-v2.pt'
CHECKPOINT_SHA256 = '2b75ac2d3c80a2644c7df5c0ec5fab9c10086ff2fa072177868e7e6c3091fe60'
CONTEXT_CHECKPOINTS = (
    (CHECKPOINT.parent / 'accompaniment-context-expanded.npz',
     'd06bbe84ee9ad251960969e0097b217e340d52562ecf178295d7c4ae57c6ee8d'),
    (CHECKPOINT.parent / 'accompaniment-context-original.npz',
     '381c150b19bdb223fc9e85f88b78dd69892b0de5c7591189967f30b59a26a8e7'),
)
RESIDUAL_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-residual-v4.npz'
RESIDUAL_SHA256 = 'ad958557c44312c8d082b9339801ae794a7673956f098106f7fb6381fa3cd24d'


class AccompanimentVerifier:
    name = 'Accompaniment note verifier v4'

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
        self.context_models = []
        for path, digest in CONTEXT_CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise PipelineError('The accompaniment context model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    context = ContextNoteModel(arrays)
                if context.threshold != .01:
                    raise ValueError('Unexpected context threshold')
                self.context_models.append(context)
            except (ValueError, KeyError, OSError) as exc:
                raise PipelineError('The accompaniment context model is invalid. Rebuild the backend and retry.') from exc
        if (not RESIDUAL_CHECKPOINT.is_file()
                or hashlib.sha256(RESIDUAL_CHECKPOINT.read_bytes()).hexdigest() != RESIDUAL_SHA256):
            raise PipelineError('The residual note model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(RESIDUAL_CHECKPOINT, allow_pickle=False) as arrays:
                self.residual_model = ContextNoteModel(arrays)
            if self.residual_model.threshold != .0375:
                raise ValueError('Unexpected residual threshold')
        except (ValueError, KeyError, OSError) as exc:
            raise PipelineError('The residual note model is invalid. Rebuild the backend and retry.') from exc

    def filter(self, path, acoustic, midi):
        """Keep original pitch, timing and velocity; reject unsupported events only."""
        import torch

        notes = [n for part in midi.instruments for n in part.notes]
        if not notes:
            return 0
        samples, rate = sf.read(path, dtype='float32')
        if samples.ndim != 1 or rate != 22050 or not 0 < len(samples) <= 33 * rate:
            raise PipelineError('Accompaniment verification needs a normalized window of at most 33 seconds.')
        x = note_features(samples, rate, acoustic, notes, include_context=True)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(torch.from_numpy((x[:, :len(FEATURE_NAMES)] - self.mean) / self.scale))).numpy()
        if not np.isfinite(probability).all():
            raise PipelineError('The accompaniment model returned invalid note confidence.')
        bounded = decode_candidates(acoustic)
        events = [[n.start, n.end, n.pitch, n.velocity] for n in notes]
        bounded_events = [[n.start, n.end, n.pitch, n.velocity]
                          for part in bounded.instruments for n in part.notes]
        context = np.asarray([model.probability(x) for model in self.context_models])
        shared = shared_events(events, bounded_events)
        keep = correction_keep(probability, context, shared,
                               threshold=self.threshold, prune=self.context_models[0].threshold)
        eligible = keep & shared & (probability <= .5)
        residual = np.ones(len(notes))
        if np.any(eligible):
            residual[eligible] = self.residual_model.probability(x[eligible])
        keep = residual_keep(keep, probability, shared, residual, threshold=self.residual_model.threshold)
        index = 0
        for part in midi.instruments:
            size = len(part.notes)
            part.notes = [note for note, accepted in zip(part.notes, keep[index:index + size], strict=True) if accepted]
            index += size
        return int(np.sum(~keep))
