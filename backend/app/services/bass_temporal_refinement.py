"""V16 refinement on V15 bass survivors; retained notes keep their identity."""

import hashlib
from pathlib import Path

import numpy as np

from app.models import PipelineError
from app.services.bass_harmonic import CHECKPOINTS
from app.services.bass_harmonic import VERSION as GUARDIAN_VERSION
from app.services.bass_temporal import BassTemporal
from app.services.note_context_model import ContextNoteModel
from app.services.temporal_note_model import (
    FEATURE_COUNT,
    VERSION,
    create_model,
    probability,
)

ASSET = Path(__file__).resolve().parents[1] / 'assets/bass-temporal-refinement-v1.npz'
EXPECTED_SHA = '2db9b76e75c353d6d5070df725de0d79860a459559fa60b514aeaff437691775'


class RefinedTemporalCandidateModel:
    def __init__(self, saved):
        import torch

        if (str(saved['version']) != VERSION or int(saved['format_version']) != 1
                or int(saved['width']) != 48 or float(saved['threshold']) != .2
                or float(saved['guardian_threshold']) != .3):
            raise ValueError('Incompatible temporal checkpoint contract')
        self.threshold = .2
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


class BassTemporalRefinement(BassTemporal):
    name = 'Bass temporal refinement v1 (expanded piano training)'

    def __init__(self):
        guardian_path, guardian_sha, names, threshold = CHECKPOINTS[1]
        for path, expected in ((ASSET, EXPECTED_SHA), (guardian_path, guardian_sha)):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass refinement model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(ASSET, allow_pickle=False) as saved:
                self.model = RefinedTemporalCandidateModel(saved)
            with np.load(guardian_path, allow_pickle=False) as saved:
                self.guardian = ContextNoteModel(saved, feature_names=names, feature_version=GUARDIAN_VERSION)
            if self.guardian.threshold != threshold or threshold != self.model.guardian_threshold:
                raise ValueError('Changed temporal guardian threshold')
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError('The bass refinement model contains incompatible data. Rebuild the backend and retry.') from exc

