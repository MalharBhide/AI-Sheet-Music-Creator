"""Pinned V8 baseline for new experiments; historical V7 helpers stay reproducible."""

import hashlib

import numpy as np
from app.services.accompaniment_verifier import (
    BROAD_CHECKPOINT,
    BROAD_GUARDIAN_CHECKPOINT,
    BROAD_GUARDIAN_SHA256,
    BROAD_SHA256,
)
from app.services.left_baseline_evidence import (
    ACOUSTIC_EVIDENCE_NAMES,
    EVIDENCE_VERSION,
    RELATION_EVIDENCE_NAMES,
    acoustic_evidence,
)
from app.services.note_context_model import ContextNoteModel, residual_keep
from cache_note_verifier import targets
from train_left_consensus_v8 import hashes as v7_hashes
from train_left_consensus_v8 import prepare as prepare_v7


def hashes():
    result = v7_hashes()
    for path, expected in ((BROAD_CHECKPOINT, BROAD_SHA256),
                           (BROAD_GUARDIAN_CHECKPOINT, BROAD_GUARDIAN_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen website V8 baseline changed')
        result[path.name] = expected
    return result


def advance(item, relation, guardian):
    decision = np.where((relation < .005) & (guardian < .05), 0., 1.)
    item['keep'] = residual_keep(item['keep'], item['p'], item['shared'], decision,
                                 threshold=.005, ceiling=1.)
    item['baseline_keep'] = item['keep'].copy()
    y, mask = targets(item['reference'], item['events'][item['keep']])
    item['y'], item['mask'] = np.zeros(len(item['events'])), np.zeros(len(item['events']), dtype=bool)
    item['y'][item['keep']], item['mask'][item['keep']] = y, mask
    return item


def prepare(directory, items):
    hashes()
    models = []
    for path, names, threshold in ((BROAD_CHECKPOINT, RELATION_EVIDENCE_NAMES, .005),
                                   (BROAD_GUARDIAN_CHECKPOINT, ACOUSTIC_EVIDENCE_NAMES, .05)):
        with np.load(path, allow_pickle=False) as saved:
            model = ContextNoteModel(saved, feature_names=names, feature_version=EVIDENCE_VERSION)
        if model.threshold != threshold:
            raise ValueError('Frozen website V8 threshold changed')
        models.append(model)
    prepared = prepare_v7(directory, items, baseline_evidence=True, full_register=True)
    return [advance(item, models[0].probability(item['x']),
                    models[1].probability(acoustic_evidence(item['x']))) for item in prepared]
