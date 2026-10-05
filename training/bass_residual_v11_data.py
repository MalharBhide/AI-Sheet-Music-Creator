"""Label-free residual features over verified, duration-correct V11 caches."""

import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
from app.services.candidate_relations import RELATION_NAMES, relation_features
from bass_training_data import NAMES as BASE_NAMES
from bass_training_data import identity, load
from cache_note_verifier import targets
from current_bass_v11_baseline import VERSION as BASELINE_VERSION
from current_bass_v11_baseline import hashes
from pitch_support_targets import keep_targets
from prepare_robust_training_stems import digest

VERSION = 'bass-v11-residual-relations-v1'
NAMES = BASE_NAMES + ('post_merge_consensus_mean',) + RELATION_NAMES[1:] + ('post_merge_context_confidence', 'post_merge_guardian_confidence')
ACOUSTIC_NAMES = BASE_NAMES[:26] + NAMES[-2:]


def features(events, x, context, guardian):
    context, guardian = np.asarray(context), np.asarray(guardian)
    if any(p.shape != (len(events),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
           for p in (context, guardian)):
        raise ValueError('Invalid V11 post-merge confidence')
    relation = relation_features(events, x, (context + guardian) * .5)
    return np.column_stack((relation, context, guardian)).astype(np.float32)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid V11 residual evidence')
    return np.column_stack((x[:, :26], x[:, -2:])).astype(np.float32)


def annotate(item):
    events = item['events']
    y, mask, promoted = keep_targets(item['pitch_reference'], events)
    actual, _ = targets(item['reference'], events)
    y[actual == 1], mask[actual == 1] = 1., True
    # Offset-aware matching can assign a different candidate than attack-only
    # one-to-one matching. Both already correct assignments are protected.
    held = mir_eval.transcription.match_notes(
        item['reference'][:, :2], librosa.midi_to_hz(item['reference'][:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]),
        onset_tolerance=.05, offset_ratio=.2, offset_min_tolerance=.05)
    held_indices = [index for _, index in held]
    y[held_indices], mask[held_indices] = 1., True
    return {**item, 'x': features(events, item['x'], item['context'], item['guardian']),
            'y': y, 'mask': mask, 'promoted': promoted,
            'protected_offset_assignments': len(held),
            'baseline_keep': np.ones(len(events), bool), 'keep': np.ones(len(events), bool)}


def load_cached(directory, groups):
    manifest = json.loads((directory / 'manifest.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if (manifest['plan_sha256'] != digest(directory / 'plan.json') or manifest['version'] != BASELINE_VERSION
            or plan['baseline_hashes'] != hashes() or not manifest['all_tests_consumed_regression']):
        raise ValueError('Changed V11 baseline cache contract')
    raw, result = {}, []
    for record in manifest['items']:
        if record['group'] not in groups:
            continue
        root, group = Path(record['root']), record['group']
        key = (root, group)
        if key not in raw:
            original_sha = (plan['fitting_manifests'] if group != 'test' else plan['consumed_regression_manifests'])[str(root)]
            if digest(root / 'manifest.json') != original_sha:
                raise ValueError('Changed V11 original source manifest')
            raw[key] = {item['id']: item for item in load(root, group)}
        item = raw[key][record['id']]
        original_manifest = json.loads((root / 'manifest.json').read_text())
        original = next(i for i in original_manifest['items'] if i['id'] == record['id'])
        if (identity(original) != record['identity']
                or original_manifest['cache_sha256'][record['id']] != record['source_cache_sha256']):
            raise ValueError('Changed V11 original source identity')
        path = directory / 'features' / (record['id'] + '.npz')
        if digest(path) != record['cache_sha256'] or digest(Path(item['audio'])) != record['audio_sha256']:
            raise ValueError('Changed V11 source audio or cached features')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['baseline_version']) != BASELINE_VERSION
                    or str(saved['plan_sha256']) != manifest['plan_sha256']
                    or str(saved['identity']) != json.dumps({k: v for k, v in record.items() if k != 'cache_sha256'}, sort_keys=True)):
                raise ValueError('Changed V11 feature identity')
            fields = ('x', 'events', 'eligible', 'context', 'guardian', 'raw_indices')
            cached = {field: saved[field] for field in fields}
        if cached['x'].shape != (len(cached['events']), 52) or not np.isfinite(cached['x']).all():
            raise ValueError('Invalid V11 duration-correct acoustic features')
        result.append(annotate({**item, **cached, 'raw_count': record['raw_candidates']}))
    return result


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if (not training or not validation
            or {i['source_group'] for i in training} & {i['source_group'] for i in validation}):
        raise ValueError('V11 residual fitting split overlaps or is empty')
    return training, validation
