"""Supervision over V14 survivors, with immutable attack/hold/coverage truth."""

import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
from bass_residual_coverage_data import protect
from bass_training_data import eligible, identity
from cache_note_verifier import targets
from current_bass_v14_baseline import VERSION as BASELINE_VERSION
from current_bass_v14_baseline import hashes
from harmonic_candidate_features import ACOUSTIC_NAMES, NAMES, acoustic_view, features
from pitch_support_targets import keep_targets
from prepare_robust_training_stems import digest

VERSION = 'bass-v14-temporal-cqt-v1'
__all__ = ['VERSION', 'NAMES', 'ACOUSTIC_NAMES', 'features', 'acoustic_view', 'annotate', 'baseline_items', 'load_cached', 'collect']


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


def baseline_items(directory, groups):
    manifest = json.loads((directory / 'manifest.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    records = manifest['items']
    if (manifest['version'] != BASELINE_VERSION or plan['version'] != BASELINE_VERSION
            or manifest['plan_sha256'] != digest(directory / 'plan.json')
            or plan['baseline_hashes'] != hashes() or not manifest['all_tests_consumed_regression']
            or not manifest['no_user_audio_or_scores'] or not plan['no_audio_inference']
            or plan['code_sha256'] != digest(Path(__file__).with_name('prepare_bass_v14_baseline.py'))
            or len({r['id'] for r in records}) != 897
            or {g: sum(r['group'] == g for r in records) for g in plan['expected_counts']} != plan['expected_counts']):
        raise ValueError('Changed V14 survivor cache contract')
    partitions = {g: {r['source_group'] for r in records if r['group'] == g} for g in plan['expected_counts']}
    if any(partitions[a] & partitions[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('V14 source partitions overlap')
    result = []
    for record in records:
        if record['group'] not in groups:
            continue
        path = directory / 'features' / (record['id'] + '.npz')
        if digest(path) != record['cache_sha256']:
            raise ValueError('Changed V14 survivor evidence')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})
                    or str(saved['plan_sha256']) != manifest['plan_sha256']):
                raise ValueError('Changed V14 survivor identity')
            item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible', 'v13_indices')}}
        if (item['base_x'].shape != (len(item['events']), 52)
                or len(item['events']) != record['retained_v14'] or not np.isfinite(item['base_x']).all()):
            raise ValueError('Invalid V14 acoustic intervals')
        np.testing.assert_array_equal(item['eligible'], eligible(item['events'], record['duration']))
        item.update(reference=np.asarray(record['reference'], float).reshape(-1, 3),
                    pitch_reference=np.asarray(record['pitch_reference'], float).reshape(-1, 3))
        result.append(annotate(item))
    return result


def load_cached(directory, groups):
    from app.services.temporal_note_model import OFFSETS, STEPS

    manifest = json.loads((directory / 'manifest.json').read_text())
    prepared = json.loads((directory / 'plan.json').read_text())
    root = Path(__file__).resolve().parents[1]
    if (manifest['plan_sha256'] != digest(directory / 'plan.json')
            or prepared['version'] != VERSION or prepared['baseline_hashes'] != hashes()
            or not manifest['no_test_audio_or_metrics'] or not manifest['no_user_audio_or_scores']
            or any(g not in ('train', 'validation') for g in groups)
            or any(digest(root / name) != sha for name, sha in prepared['code_sha256'].items())):
        raise ValueError('Changed temporal preparation or illegal fitting partition')
    parent = Path(prepared['parent_root'])
    if (digest(parent / 'manifest.json') != prepared['parent_manifest_sha256']
            or digest(parent / 'plan.json') != prepared['parent_plan_sha256']):
        raise ValueError('Changed temporal source baseline')
    items = {i['id']: i for i in baseline_items(parent, groups)}
    result = []
    for record in manifest['items']:
        if record['group'] not in groups:
            continue
        item = items.pop(record['id'])
        path = directory / 'features' / (record['id'] + '.npz')
        if (digest(path) != record['cache_sha256'] or record['source_cache_sha256'] != item['cache_sha256']
                or record['source_group'] != item['source_group'] or record['group'] != item['group']):
            raise ValueError('Changed temporal interval/source identity')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['plan_sha256']) != manifest['plan_sha256']
                    or str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})):
                raise ValueError('Changed temporal cache identity')
            np.testing.assert_array_equal(saved['events'], item['events'])
            frames = saved['frames']
        if (frames.shape != (len(item['events']), STEPS, len(OFFSETS)) or not np.isfinite(frames).all()
                or np.any((frames < 0) | (frames > 1))):
            raise ValueError('Invalid temporal envelopes')
        result.append({**item, 'frames': frames})
    if items:
        raise ValueError('Incomplete temporal fitting cache')
    return result


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if len(training) != 531 or len(validation) != 146:
        raise ValueError('Incomplete temporal fitting or selection partition')
    return training, validation
