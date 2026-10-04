"""Cache the actual balanced bass decoder; never reuse accompaniment events."""

import contextlib
import hashlib
import json
import os

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from app.services.note_evidence import (
    CONTEXT_NAMES,
    CONTEXT_VERSION,
    FEATURE_NAMES,
    note_features,
)
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict
from cache_note_verifier import targets
from pitch_support_targets import keep_targets

VERSION = 'balanced-bass-candidates-v1'
EDGE_SECONDS = 2.5
NAMES = FEATURE_NAMES + CONTEXT_NAMES
_MODEL = None


def identity(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def eligible(events, duration):
    return ((events[:, 0] >= EDGE_SECONDS) & (events[:, 1] <= duration - EDGE_SECONDS)
            & (events[:, 2] >= 21) & (events[:, 2] < 60))


def cache_one(directory, item):
    global _MODEL
    cache = directory / 'features'
    cache.mkdir(exist_ok=True)
    path = cache / (item['id'] + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['identity']) != identity(item) or str(saved['version']) != VERSION:
                raise ValueError('Changed bass candidate cache identity')
        return path
    samples, rate = librosa.load(item['audio'], sr=22050, mono=True, duration=30)
    if len(samples) < rate or not np.isfinite(samples).all():
        raise ValueError('Invalid bass waveform clock')
    if _MODEL is None:
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        acoustic, midi, _ = predict(item['audio'], model_or_model_path=_MODEL,
            minimum_frequency=float(pretty_midi.note_number_to_hz(21)),
            maximum_frequency=float(pretty_midi.note_number_to_hz(60)),
            onset_threshold=.5, frame_threshold=.3, minimum_note_length=90.,
            multiple_pitch_bends=False, melodia_trick=False)
        notes = [note for part in midi.instruments for note in part.notes]
        x = note_features(samples, rate, acoustic, notes, include_context=True)
    events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], dtype=float).reshape(-1, 4)
    duration = min(30., sf.info(item['audio']).duration)
    if duration != len(samples) / rate or np.any((events[:, 2] < 21) | (events[:, 2] >= 60)):
        raise ValueError('Bass decoder differs from production clock or frequency bounds')
    first, stop = item['evaluation_window']
    cropped = (events[:, 0] >= first) & (events[:, 0] < stop)
    events, x = events[cropped], x[cropped]
    reference = np.asarray(item['reference'], dtype=float).reshape(-1, 3)
    pitch_reference = np.asarray(item['pitch_reference'], dtype=float).reshape(-1, 3)
    attack_y, _ = targets(reference, events)
    y, mask, promoted = keep_targets(pitch_reference, events)
    # Actual attack matches are always positive even when pitch support differs.
    y[attack_y == 1], mask[attack_y == 1] = 1., True
    np.savez_compressed(path, x=x, events=events, reference=reference, pitch_reference=pitch_reference,
                        y=y, mask=mask, promoted=promoted, eligible=eligible(events, duration),
                        duration=duration, seconds=stop - first, identity=identity(item),
                        version=VERSION, feature_version=CONTEXT_VERSION)
    return path


def load(directory, group):
    manifest = json.loads((directory / 'manifest.json').read_text())
    items = []
    for item in manifest['items']:
        if item['group'] != group:
            continue
        path = directory / 'features' / (item['id'] + '.npz')
        expected = manifest['cache_sha256'][item['id']]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed frozen bass features')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['identity']) != identity(item) or str(saved['version']) != VERSION
                    or str(saved['feature_version']) != CONTEXT_VERSION):
                raise ValueError('Incompatible bass cache contract')
            fields = ('x', 'events', 'reference', 'pitch_reference', 'y', 'mask', 'eligible', 'seconds')
            result = {**item, **{key: saved[key] for key in fields}}
        if result['x'].shape != (len(result['events']), len(NAMES)) or not np.isfinite(result['x']).all():
            raise ValueError('Invalid bass context features')
        items.append(result)
    return items
