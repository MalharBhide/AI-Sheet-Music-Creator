"""Evaluate a new pre-boundary filter against the complete pinned website V9."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from app.services.accompaniment_verifier import (
    BOUNDARY_CHECKPOINT,
    BOUNDARY_GUARDIAN_CHECKPOINT,
    BOUNDARY_GUARDIAN_SHA256,
    BOUNDARY_SHA256,
)
from app.services.note_context_model import ContextNoteModel, residual_keep
from app.services.repeat_boundary import (
    ACOUSTIC_NAMES,
    RELATION_NAMES,
    VERSION,
    boundaries,
    features,
    merge,
)
from current_accompaniment_baseline import hashes as v8_hashes
from current_accompaniment_baseline import prepare as prepare_v8
from evaluate_context_correction import matched_references
from pitch_interval_coverage import compare
from train_left_hand_verifier import held_matches
from train_note_verifier import metrics
from train_residual_verifier import cached_external


def hashes():
    result = v8_hashes()
    for path, expected in ((BOUNDARY_CHECKPOINT, BOUNDARY_SHA256),
                           (BOUNDARY_GUARDIAN_CHECKPOINT, BOUNDARY_GUARDIAN_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned website V9 changed')
        result[path.name] = expected
    return result


@lru_cache(maxsize=1)
def heads():
    hashes()
    models = []
    for path, names, threshold in ((BOUNDARY_CHECKPOINT, RELATION_NAMES, .05),
                                   (BOUNDARY_GUARDIAN_CHECKPOINT, ACOUSTIC_NAMES, .025)):
        with np.load(path, allow_pickle=False) as saved:
            model = ContextNoteModel(saved, feature_names=names, feature_version=VERSION)
        if model.threshold != threshold:
            raise ValueError('Pinned website V9 threshold changed')
        models.append(model)
    return models


def project(item, keep=None):
    """Reapply the actual V9 boundary stage after a proposed note-filter decision."""
    state = {**item, 'keep': item['keep'] if keep is None else keep}
    pairs = boundaries(state)
    if not len(pairs):
        return state['events'][state['keep']].copy()
    # The pre-boundary audio/geometry is frozen. Reuse identical pair evidence
    # across threshold sweeps; new neighbor pairs still receive real inference.
    identity = hashlib.sha256(item['events'].tobytes() + item['x'].tobytes()).digest()
    if item.get('_boundary_cache_identity') != identity:
        item['_boundary_cache_identity'], item['_boundary_cache'] = identity, {}
    cache = item['_boundary_cache']
    missing = np.asarray([pair for pair in pairs if tuple(pair) not in cache], dtype=int).reshape(-1, 2)
    relation, guardian = heads()
    if len(missing):
        repeat = relation.probability(features(state, missing))
        confidence = guardian.probability(features(state, missing, acoustic=True))
        cache.update({tuple(pair): (p, g) for pair, p, g in zip(missing, repeat, confidence, strict=True)})
    repeat = np.asarray([cache[tuple(pair)][0] for pair in pairs])
    confidence = np.asarray([cache[tuple(pair)][1] for pair in pairs])
    return merge(state, pairs, repeat, confidence, relation.threshold, guardian.threshold)[0]


def prepare(directory, raw, baseline_evidence=True, full_register=True):
    if not baseline_evidence or not full_register:
        raise ValueError('The V9 pre-boundary contract requires full V8 confidence evidence')
    hashes()
    items = prepare_v8(directory, raw)
    for item in items:
        item['baseline_events'] = project(item)
    return items


def score(items, probabilities, threshold, memo=None, *, ceiling=1., preserve_coverage=True):
    if ceiling != 1. or not preserve_coverage:
        raise ValueError('V9 comparison cannot relax pitch coverage or its confidence policy')
    rows = []
    for item, probability in zip(items, probabilities, strict=True):
        before = item['baseline_events']
        accepted = residual_keep(item['keep'], item['p'], item['shared'], probability,
                                 threshold=threshold, ceiling=ceiling)
        key = (id(item), accepted.tobytes())
        if memo is not None and key in memo:
            rows.append(memo[key])
            continue
        after = before if np.array_equal(accepted, item['keep']) else project(item, accepted)
        old, new = [metrics(item['reference'], events, item['seconds']) for events in (before, after)]
        attacks = matched_references(item['reference'], before).issubset(matched_references(item['reference'], after))
        holds = held_matches(item['reference'], before).issubset(held_matches(item['reference'], after))
        pitch_reference = np.asarray(item.get('pitch_reference', item['reference']), dtype=float).reshape(-1, 3)
        coverage = compare(pitch_reference, before, after)
        row = {'id': item['id'], 'corpus': item['corpus'], 'deployed_v9': old, 'candidate': new,
               'matched_references_preserved': attacks, 'held_references_preserved': holds,
               'reference_pitch_coverage_preserved': coverage['passes'],
               'lost_reference_pitch_seconds': coverage['lost_reference_seconds'],
               'passes': attacks and holds and coverage['passes']
               and new['false_positives'] <= old['false_positives']}
        rows.append(row)
        if memo is not None:
            memo[key] = row
    return {'threshold': threshold, 'passes': all(row['passes'] for row in rows),
            'false_notes_removed': sum(row['deployed_v9']['false_positives'] - row['candidate']['false_positives']
                                       for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']], 'per_recording': rows}


def regression_items(directory):
    from train_left_consensus_v8 import regression_items as previous

    items = previous(directory)
    for name in ('relational-piano-fresh-120', 'repeat-boundary-stress-v1'):
        items.extend(cached_external(directory, directory.parent / name / 'manifest.json', directory))
    return items


def contract_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ['training/accompaniment_baseline_v9.py', 'training/current_accompaniment_baseline.py',
             'backend/app/services/repeat_boundary.py', 'backend/app/services/candidate_relations.py',
             'backend/app/services/left_baseline_evidence.py', 'backend/app/services/note_context_model.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
