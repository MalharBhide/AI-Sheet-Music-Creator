"""Label-free context over actual V12 notes, with unchanged pitch-support truth."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_residual import NAMES as RESIDUAL_NAMES
from app.services.bass_residual import features as residual_features
from bass_residual_coverage_data import protect
from bass_residual_v11_data import annotate as annotate_v11
from bass_training_data import identity
from current_bass_v12_baseline import VERSION as BASELINE_VERSION
from current_bass_v12_baseline import hashes
from prepare_robust_training_stems import digest

VERSION = 'bass-v12-retained-context-v1'
NAMES = RESIDUAL_NAMES + ('released_residual_context_confidence', 'released_residual_guardian_confidence')
ACOUSTIC_NAMES = NAMES[:26] + NAMES[-4:]


def features(events, base_x, context, guardian, residual_context, residual_guardian):
    for confidence in (residual_context, residual_guardian):
        confidence = np.asarray(confidence)
        if (confidence.shape != (len(events),) or not np.isfinite(confidence).all()
                or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError('Invalid V12 residual confidence')
    relation = residual_features(events, base_x, context, guardian)
    return np.column_stack((relation, residual_context, residual_guardian)).astype(np.float32)


def acoustic_view(x):
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid V12 retained feature contract')
    return np.column_stack((x[:, :26], x[:, -4:])).astype(np.float32)


def annotate(item):
    protected = protect(annotate_v11({**item, 'x': item['base_x']}))
    return {**protected, 'x': features(item['events'], item['base_x'], item['context'], item['guardian'],
                                    item['residual_context'], item['residual_guardian'])}


def load_cached(directory, groups):
    manifest = json.loads((directory / 'manifest.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if (manifest['version'] != BASELINE_VERSION or plan['version'] != BASELINE_VERSION
            or manifest['plan_sha256'] != digest(directory / 'plan.json')
            or plan['baseline_hashes'] != hashes() or not manifest['all_tests_consumed_regression']
            or not manifest['no_user_audio_or_scores'] or not plan['no_audio_inference']
            or plan['code_sha256'] != digest(Path(__file__).with_name('prepare_bass_v12_baseline.py'))):
        raise ValueError('Changed V12 retained cache contract')
    result = []
    for record in manifest['items']:
        if record['group'] not in groups:
            continue
        path = directory / 'features' / (record['id'] + '.npz')
        if digest(path) != record['cache_sha256']:
            raise ValueError('Changed V12 retained evidence')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})
                    or str(saved['plan_sha256']) != manifest['plan_sha256']):
                raise ValueError('Changed V12 retained identity')
            item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible', 'context',
                    'guardian', 'residual_context', 'residual_guardian', 'v11_indices')}}
        if (item['base_x'].shape != (len(item['events']), 52)
                or len(item['events']) != record['retained_v12']
                or not np.isfinite(item['base_x']).all()):
            raise ValueError('Invalid V12 acoustic intervals')
        item.update(reference=np.asarray(record['reference'], float).reshape(-1, 3),
                    pitch_reference=np.asarray(record['pitch_reference'], float).reshape(-1, 3))
        result.append(annotate(item))
    return result


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if (not training or not validation
            or {i['source_group'] for i in training} & {i['source_group'] for i in validation}):
        raise ValueError('V12 source partitions overlap or are empty')
    return training, validation
