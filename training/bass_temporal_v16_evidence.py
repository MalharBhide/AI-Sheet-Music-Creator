"""Verify and read every frozen V16 fitting/regression/first-pass input."""

import json
from pathlib import Path

import numpy as np
from bass_temporal_v15_data import baseline_items, piano_items
from bass_temporal_v16_release import load_frozen
from bass_training_data import eligible
from fresh_bass_temporal_v16 import require_regression
from prepare_robust_training_stems import digest
from train_bass_temporal_v16 import FRESH_SEEDS


def cached_items(folder, manifest):
    items = []
    for record in manifest['items']:
        path = folder / record['id'] / 'features.npz'
        if (record['group'] != 'test' or digest(path) != record['cache_sha256']
                or digest(Path(record['audio'])) != record['audio_sha256']):
            raise ValueError('Changed V16 regression audio/cache')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != manifest['plan_sha256']:
                raise ValueError('Changed V16 regression feature identity')
            item = {**record, **{key: saved[key] for key in ('events', 'base_x', 'x', 'frames', 'eligible')}}
        n = len(item['events'])
        if (item['events'].shape != (n, 4) or item['base_x'].shape != (n, 52)
                or item['x'].shape != (n, 84) or item['frames'].shape != (n, 40, 9)
                or any(not np.isfinite(item[key]).all() for key in ('events', 'base_x', 'x', 'frames'))):
            raise ValueError('Invalid V16 regression evidence')
        np.testing.assert_array_equal(item['eligible'], eligible(item['events'], record['duration']))
        items.append(item)
    return items


def fresh_items(run, fresh, winner):
    result = json.loads((run / 'original-first-pass.json').read_text())
    manifest = json.loads((fresh / 'manifest.json').read_text())
    plan = json.loads((fresh / 'plan.json').read_text())
    if (not result['passes'] or result['false_notes_removed'] <= 0
            or len(result['per_recording']) != 32 or any(not r['passes'] for r in result['per_recording'])
            or not result['first_pass_complete'] or not result['now_consumed_regression'] or not result['no_retuning']
            or result['checkpoint_sha256'] != winner['checkpoint_sha256']
            or result['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or result['test_manifest_sha256'] != digest(fresh / 'manifest.json')
            or (result['threshold'], result['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])
            or manifest['plan_sha256'] != digest(fresh / 'plan.json') or plan['seeds'] != list(FRESH_SEEDS)
            or not plan['positive_first_pass_gain_required'] or not manifest['now_consumed_regression']
            or not manifest['no_fitting_selection_user_audio_or_scores']
            or plan['checkpoint_sha256'] != winner['checkpoint_sha256']
            or plan['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or plan['consumed_regression_sha256'] != digest(run / 'consumed-regression.json')
            or plan['piano_regression_sha256'] != digest(run / 'piano-regression.json')):
        raise ValueError('Failed or changed positive V16 first-pass evidence')
    root = Path(__file__).resolve().parents[1]
    if any(digest(root / name) != expected for name, expected in plan['code_sha256'].items()):
        raise ValueError('Changed V16 first-pass preparation')
    expected = ['bass-temporal-v16-fresh-v1-' + str(seed) for seed in FRESH_SEEDS]
    if ([r['id'] for r in manifest['items']] != expected
            or [r['id'] for r in result['per_recording']] != expected
            or [r['source_group'] for r in manifest['items']] != ['original-bass-seed-' + str(seed) for seed in FRESH_SEEDS]):
        raise ValueError('Changed V16 first-pass source coverage')
    return cached_items(fresh, manifest)


def all_items(run, fresh):
    plan, winner, model, normalizer = load_frozen(run)
    require_regression(run, winner)
    baseline = baseline_items(Path(plan['baseline_root']), ('train', 'validation', 'test'))
    piano = piano_items(Path(plan['piano_root']), ('train', 'validation'))
    reserved = run.parent / 'v16-piano-regression-v1'
    reserved_items = cached_items(reserved, json.loads((reserved / 'manifest.json').read_text()))
    first = fresh_items(run, fresh, winner)
    items = baseline + piano + reserved_items + first
    if (len(baseline) != 929 or len(piano) != 56 or len(reserved_items) != 12 or len(first) != 32
            or len({i['id'] for i in items}) != 1029):
        raise ValueError('Incomplete V16 fitting/validation/regression evidence')
    return plan, winner, model, normalizer, items
