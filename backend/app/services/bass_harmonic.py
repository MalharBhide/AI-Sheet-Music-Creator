"""Acoustic-only rejection of unsupported notes after the released V13 route."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.bass_texture import BASE_NAMES
from app.services.bass_texture import NAMES as SALIENCE_NAMES
from app.services.bass_texture import features as salience_features
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import note_features

VERSION = 'bass-v13-harmonic-relations-v1'
ASSETS = Path(__file__).resolve().parents[1] / 'assets'

INTERVALS = (12, 19, 24)
HARMONIC_NAMES = tuple(name + '_' + str(interval) for interval in INTERVALS
                       for name in ('lower_attack_strength', 'harmonic_attack_difference',
                                    'harmonic_attack_alignment', 'harmonic_release_alignment'))
NAMES = SALIENCE_NAMES + HARMONIC_NAMES
ACOUSTIC_NAMES = BASE_NAMES
CHECKPOINTS = (
    (ASSETS / 'bass-harmonic-v1.npz',
     '8184a7c4009d2e26bc1d2f264ec58d8503182cda79667b9c3af7a8bb856ce66c', NAMES, .025),
    (ASSETS / 'bass-harmonic-guardian-v1.npz',
     'bcab8b156e54db37c3c19dd74e553106499722503abb043ba7fb491d3418f76e', ACOUSTIC_NAMES, .3),
)


def features(events, base_x):
    events = np.asarray(events, float)
    base_x = np.asarray(base_x, np.float32)
    x = salience_features(events, base_x)
    if not len(events):
        return np.empty((0, len(NAMES)), np.float32)
    starts, ends, pitches, _ = events.T
    attacks = base_x[:, BASE_NAMES.index('onset_at_attack')]
    strength = base_x[:, BASE_NAMES.index('note_mean')]
    rows = []
    for index, (start, end, pitch, _) in enumerate(events):
        neighborhood = (np.abs(starts - start) <= 2.) & (np.arange(len(events)) != index)
        overlaps = (np.minimum(end, ends) - np.maximum(start, starts)) > 0
        row = []
        for interval in INTERVALS:
            lower = neighborhood & overlaps & (pitches == pitch - interval)
            maximum = float(np.max(attacks[lower], initial=0.))
            attack_alignment = attacks[lower] * np.exp(-np.abs(starts[lower] - start) / .12)
            release_alignment = strength[lower] * np.exp(-np.abs(ends[lower] - end) / .12)
            row.extend((maximum, float(np.clip(attacks[index] - maximum, -1., 1.)),
                        float(np.max(attack_alignment, initial=0.)),
                        float(np.max(release_alignment, initial=0.))))
        rows.append(row)
    result = np.column_stack((x, np.asarray(rows, np.float32))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Invalid harmonic candidate evidence')
    return result


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid V13 harmonic feature contract')
    return x[:, :len(BASE_NAMES)]


class BassHarmonic:
    name = 'Bass harmonic verifier v1 (trained attack/release rejection)'

    def __init__(self):
        if len(CHECKPOINTS) != 2:
            raise PipelineError('The bass harmonic model has not been released.')
        self.models = []
        for path, expected, names, threshold in CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass harmonic model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as saved:
                    model = ContextNoteModel(saved, feature_names=names, feature_version=VERSION)
                if model.threshold != threshold:
                    raise ValueError('Changed bass harmonic threshold')
            except (KeyError, ValueError, OSError) as exc:
                raise PipelineError('The bass harmonic model contains incompatible data. Rebuild the backend and retry.') from exc
            self.models.append(model)

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Bass harmonic verification needs finite mono audio at 22050 Hz.')
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
