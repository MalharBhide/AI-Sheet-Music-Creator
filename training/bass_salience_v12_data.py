"""Candidate salience from observed acoustics, without fitted confidence inputs."""

import numpy as np
from app.services.candidate_relations import RELATION_NAMES, relation_features
from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES
from bass_residual_v12_data import load_cached as retained

VERSION = 'bass-v12-acoustic-salience-v2'
BASE_NAMES = FEATURE_NAMES + CONTEXT_NAMES
NAMES = BASE_NAMES + ('observed_acoustic_salience',) + RELATION_NAMES[1:]
ACOUSTIC_NAMES = BASE_NAMES


def features(events, base_x):
    # These fixed, pitch-relative observed channels never contain label truth
    # or in-sample confidence from a previously fitted rejection classifier.
    if base_x.shape != (len(events), len(BASE_NAMES)) or not np.isfinite(base_x).all():
        raise ValueError('Invalid acoustic salience evidence')
    salience = np.clip(np.mean(base_x[:, [BASE_NAMES.index(name) for name in
                       ('note_mean', 'onset_at_attack', 'relative_note')]], axis=1), 0., 1.)
    return relation_features(events, base_x, salience)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid acoustic salience contract')
    return x[:, :len(BASE_NAMES)]


def load_cached(directory, groups):
    return [{**item, 'x': features(item['events'], item['base_x'])} for item in retained(directory, groups)]


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if (not training or not validation
            or {i['source_group'] for i in training} & {i['source_group'] for i in validation}):
        raise ValueError('Acoustic salience source partitions overlap or are empty')
    return training, validation
