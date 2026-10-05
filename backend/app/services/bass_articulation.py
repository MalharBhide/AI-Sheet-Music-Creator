"""Portable learned repair of split bass holds, preserving rests and source parts."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES, note_features
from app.services.repeat_boundary import merge

VERSION = 'bass-key-articulation-pairs-v1'
GEOMETRY_NAMES = ('attack_spacing', 'boundary_gap', 'before_duration', 'after_duration')
NOTE_NAMES = FEATURE_NAMES + CONTEXT_NAMES + ('v10_context_confidence', 'v10_guardian_confidence')
ACOUSTIC_NOTE_NAMES = FEATURE_NAMES + ('v10_context_confidence', 'v10_guardian_confidence')
RELATION_NAMES = tuple('before_' + n for n in NOTE_NAMES) + tuple('after_' + n for n in NOTE_NAMES) + GEOMETRY_NAMES
ACOUSTIC_NAMES = tuple('before_' + n for n in ACOUSTIC_NOTE_NAMES) + tuple('after_' + n for n in ACOUSTIC_NOTE_NAMES) + GEOMETRY_NAMES
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
CHECKPOINTS = (
    (ASSETS / 'bass-articulation-v1.npz', 'dec1bc16a2a41165e9c809fc8b11712f0392464f240ef6477b9c2d7195a6cd59', RELATION_NAMES),
    (ASSETS / 'bass-articulation-guardian-v1.npz', '101556c153e6b34688e6003dd4178152522290f3543fdceeeeb8aef1d8d3dc51', ACOUSTIC_NAMES),
)


def boundaries(item):
    events, previous, pairs = item['events'], {}, []
    parts = item.get('parts', np.zeros(len(events), dtype=int))
    for index in np.argsort(events[:, 0], kind='stable'):
        if not item['keep'][index]:
            continue
        start, _, pitch, _ = events[index]
        key = (int(parts[index]), int(pitch))
        earlier = previous.get(key)
        previous[key] = int(index)
        if earlier is None or not (item['shared'][earlier] and item['shared'][index]):
            continue
        gap = start - events[earlier, 1]
        if -.03 <= gap <= 0 and .15 <= start - events[earlier, 0] <= 2.:
            pairs.append((earlier, int(index)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def features(item, pairs, acoustic=False):
    x = item['x'][:, :26] if acoustic else item['x']
    x = np.column_stack((x, item['baseline_p'], item['baseline_g']))
    before, after = pairs[:, 0], pairs[:, 1]
    events = item['events']
    geometry = np.column_stack((events[after, 0] - events[before, 0],
                               events[after, 0] - events[before, 1],
                               events[before, 1] - events[before, 0],
                               events[after, 1] - events[after, 0]))
    return np.column_stack((x[before], x[after], geometry)).astype(np.float32)


class BassArticulation:
    name = 'Bass articulation v1 (trained key-release protection)'

    def __init__(self):
        if len(CHECKPOINTS) != 2:
            raise PipelineError('The bass articulation model has not been released.')
        self.models = []
        for path, expected, names in CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise PipelineError('The bass articulation model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as saved:
                    model = ContextNoteModel(saved, feature_names=names, feature_version=VERSION)
                if model.threshold != .05:
                    raise ValueError('Changed bass articulation threshold')
            except (KeyError, ValueError, OSError) as exc:
                raise PipelineError('The bass articulation model contains incompatible data. Rebuild the backend and retry.') from exc
            self.models.append(model)

    def filter(self, path, acoustic, midi, baseline):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Bass articulation needs finite mono audio at 22050 Hz.')
        x = note_features(waveform, rate, acoustic, notes, include_context=True)
        events = np.asarray([[note.start, note.end, note.pitch, note.velocity] for note in notes], dtype=float)
        p, g = [model.probability(x[:, :model.feature_count]) for model in baseline.models]
        duration = len(waveform) / rate
        shared = (events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5) & (events[:, 2] >= 21) & (events[:, 2] < 60)
        item = {'events': events, 'x': x, 'baseline_p': p, 'baseline_g': g, 'shared': shared,
                'keep': np.ones(len(events), dtype=bool),
                'parts': np.asarray([index for index, part in enumerate(midi.instruments) for _ in part.notes])}
        pairs = boundaries(item)
        if not len(pairs):
            return 0
        repeat, guardian = [model.probability(features(item, pairs, acoustic=acoustic_view))
                            for model, acoustic_view in zip(self.models, (False, True), strict=True)]
        _, merged = merge(item, pairs, repeat, guardian, .05, .05)
        owner = np.arange(len(notes))
        removed = set()
        for before, after in merged:
            first, second = int(owner[before]), int(owner[after])
            notes[first].end = max(notes[first].end, notes[second].end)
            removed.add(id(notes[second]))
            owner[owner == second] = first
        for part in midi.instruments:
            part.notes = [note for note in part.notes if id(note) not in removed]
        return len(merged)
