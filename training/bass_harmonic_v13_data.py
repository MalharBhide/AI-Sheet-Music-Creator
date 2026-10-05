"""Supervision over V13 survivors, with immutable attack/hold/coverage truth."""

import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
from bass_residual_coverage_data import protect
from bass_training_data import eligible, identity
from cache_note_verifier import targets
from current_bass_v13_baseline import VERSION as BASELINE_VERSION
from current_bass_v13_baseline import hashes
from harmonic_candidate_features import ACOUSTIC_NAMES, NAMES, acoustic_view, features
from pitch_support_targets import keep_targets
from prepare_robust_training_stems import digest

VERSION = 'bass-v13-harmonic-relations-v1'
__all__ = ['VERSION', 'NAMES', 'ACOUSTIC_NAMES', 'features', 'acoustic_view', 'annotate', 'load_cached', 'collect']


def annotate(item):
    events = item['events']
    y, mask, promoted = keep_targets(item['pitch_reference'], events)
    actual, _ = targets(item['reference'], events)
    y[actual == 1], mask[actual == 1] = 1., True
    held = mir_eval.transcription.match_notes(
        item['reference'][:, :2], librosa.midi_to_hz(item['reference'][:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]),
        onset_tolerance=.05, offset_ratio=.2, offset_min_tolerance=.05)
    held_indices = [index for _, index in held]
    y[held_indices], mask[held_indices] = 1., True
    return protect({**item, 'x': features(events, item['base_x']), 'y': y, 'mask': mask,
                    'promoted': promoted, 'protected_offset_assignments': len(held),
                    'baseline_keep': np.ones(len(events), bool), 'keep': np.ones(len(events), bool)})


def load_cached(directory, groups):
    manifest = json.loads((directory / 'manifest.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    records = manifest['items']
    if (manifest['version'] != BASELINE_VERSION or plan['version'] != BASELINE_VERSION
            or manifest['plan_sha256'] != digest(directory / 'plan.json')
            or plan['baseline_hashes'] != hashes() or not manifest['all_tests_consumed_regression']
            or not manifest['no_user_audio_or_scores'] or not plan['no_audio_inference']
            or plan['code_sha256'] != digest(Path(__file__).with_name('prepare_bass_v13_baseline.py'))
            or len({r['id'] for r in records}) != 865
            or {g: sum(r['group'] == g for r in records) for g in plan['expected_counts']} != plan['expected_counts']):
        raise ValueError('Changed V13 survivor cache contract')
    partitions = {g: {r['source_group'] for r in records if r['group'] == g} for g in plan['expected_counts']}
    if any(partitions[a] & partitions[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('V13 source partitions overlap')
    result = []
    for record in records:
        if record['group'] not in groups:
            continue
        path = directory / 'features' / (record['id'] + '.npz')
        if digest(path) != record['cache_sha256']:
            raise ValueError('Changed V13 survivor evidence')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})
                    or str(saved['plan_sha256']) != manifest['plan_sha256']):
                raise ValueError('Changed V13 survivor identity')
            item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible', 'v12_indices')}}
        if (item['base_x'].shape != (len(item['events']), 52)
                or len(item['events']) != record['retained_v13'] or not np.isfinite(item['base_x']).all()):
            raise ValueError('Invalid V13 acoustic intervals')
        np.testing.assert_array_equal(item['eligible'], eligible(item['events'], record['duration']))
        item.update(reference=np.asarray(record['reference'], float).reshape(-1, 3),
                    pitch_reference=np.asarray(record['pitch_reference'], float).reshape(-1, 3))
        result.append(annotate(item))
    return result


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if len(training) != 531 or len(validation) != 146:
        raise ValueError('Incomplete V13 fitting or selection partition')
    return training, validation
