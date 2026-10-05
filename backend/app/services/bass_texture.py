"""Acoustic-only rejection of unsupported notes after the released V12 route."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.candidate_relations import RELATION_NAMES, relation_features
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES, note_features

VERSION = 'bass-v12-acoustic-texture-v3'
BASE_NAMES = FEATURE_NAMES + CONTEXT_NAMES
NAMES = BASE_NAMES + ('observed_acoustic_salience',) + RELATION_NAMES[1:]
ACOUSTIC_NAMES = BASE_NAMES
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
CHECKPOINTS = (
    (ASSETS / 'bass-texture-v1.npz',
     'f1e829c7d4328afc38268cc8dbe8afb526c6860efafd3d272170cf3aca768d3e', NAMES, .025),
    (ASSETS / 'bass-texture-guardian-v1.npz',
     '82f4ed13e9cf5a7a832a8a066f23ff9f7642aa187da664779a900d55b807e994', ACOUSTIC_NAMES, .3),
)


def features(events, base_x):
    if base_x.shape != (len(events), len(BASE_NAMES)) or not np.isfinite(base_x).all():
        raise ValueError('Invalid acoustic salience evidence')
    salience = np.clip(np.mean(base_x[:, [BASE_NAMES.index(name) for name in
                       ('note_mean', 'onset_at_attack', 'relative_note')]], axis=1), 0., 1.)
    return relation_features(events, base_x, salience)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid acoustic salience contract')
    return x[:, :len(BASE_NAMES)]


class BassTexture:
    name = 'Bass texture verifier v1 (trained acoustic rejection)'

    def __init__(self):
        if len(CHECKPOINTS) != 2:
            raise PipelineError('The bass texture model has not been released.')
        self.models = []
        for path, expected, names, threshold in CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass texture model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as saved:
                    model = ContextNoteModel(saved, feature_names=names, feature_version=VERSION)
                if model.threshold != threshold:
                    raise ValueError('Changed bass texture threshold')
            except (KeyError, ValueError, OSError) as exc:
                raise PipelineError('The bass texture model contains incompatible data. Rebuild the backend and retry.') from exc
            self.models.append(model)

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Bass texture verification needs finite mono audio at 22050 Hz.')
        base_x = note_features(waveform, rate, acoustic, notes, include_context=True)
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        x = features(events, base_x)
        p, g = [model.probability(view) for model, view in zip(self.models, (x, acoustic_view(x)), strict=True)]
        duration = len(waveform) / rate
        eligible = (events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5) & (events[:, 2] >= 21) & (events[:, 2] < 60)
        keep = ~(eligible & (p < self.models[0].threshold) & (g < self.models[1].threshold))
        accepted = {id(n) for n, k in zip(notes, keep, strict=True) if k}
        for part in midi.instruments:
            part.notes = [n for n in part.notes if id(n) in accepted]
        return int((~keep).sum())
