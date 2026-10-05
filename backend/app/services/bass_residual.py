"""Portable residual rejection of unsupported bass pitches after V11 hold repair."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.candidate_relations import RELATION_NAMES, relation_features
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES, note_features

VERSION = 'bass-v11-pitch-coverage-residual-v2'
NAMES = FEATURE_NAMES + CONTEXT_NAMES + ('post_merge_consensus_mean',) + RELATION_NAMES[1:] + ('post_merge_context_confidence', 'post_merge_guardian_confidence')
ACOUSTIC_NAMES = FEATURE_NAMES + NAMES[-2:]
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
CHECKPOINTS = (
    (ASSETS / 'bass-residual-v1.npz', '0362480f1210e4f7220fdd6984cee71260fd0145f87559710789bdfabc319988', NAMES, .05),
    (ASSETS / 'bass-residual-guardian-v1.npz', 'cc09f098276d7c31cbafef90fdf6f18f42424a7b4159270bc2e3c332372b9dfb', ACOUSTIC_NAMES, .1),
)


def features(events, x, context, guardian):
    context, guardian = np.asarray(context), np.asarray(guardian)
    if any(p.shape != (len(events),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
           for p in (context, guardian)):
        raise ValueError('Invalid post-merge bass confidence')
    relation = relation_features(events, x, (context + guardian) * .5)
    return np.column_stack((relation, context, guardian)).astype(np.float32)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid residual bass evidence')
    return np.column_stack((x[:, :26], x[:, -2:])).astype(np.float32)


class BassResidual:
    name = 'Bass residual v1 (trained unsupported-pitch rejection)'

    def __init__(self):
        if len(CHECKPOINTS) != 2:
            raise PipelineError('The residual bass model has not been released.')
        self.models = []
        for path, expected, names, threshold in CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The residual bass model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as saved:
                    model = ContextNoteModel(saved, feature_names=names, feature_version=VERSION)
                if model.threshold != threshold:
                    raise ValueError('Changed residual bass threshold')
            except (KeyError, ValueError, OSError) as exc:
                raise PipelineError('The residual bass model contains incompatible data. Rebuild the backend and retry.') from exc
            self.models.append(model)

    def filter(self, path, acoustic, midi, baseline):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Residual bass verification needs finite mono audio at 22050 Hz.')
        x = note_features(waveform, rate, acoustic, notes, include_context=True)
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], dtype=float)
        p, g = [model.probability(x[:, :model.feature_count]) for model in baseline.models]
        x = features(events, x, p, g)
        p, g = [model.probability(view) for model, view in zip(self.models, (x, acoustic_view(x)), strict=True)]
        duration = len(waveform) / rate
        eligible = (events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5) & (events[:, 2] >= 21) & (events[:, 2] < 60)
        keep = ~(eligible & (p < self.models[0].threshold) & (g < self.models[1].threshold))
        accepted = {id(n) for n, keep_note in zip(notes, keep, strict=True) if keep_note}
        for part in midi.instruments:
            part.notes = [n for n in part.notes if id(n) in accepted]
        return int((~keep).sum())
