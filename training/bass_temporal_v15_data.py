"""Expanded supervised CNN data, bound to actual shipped V15 survivors."""

import json
from pathlib import Path

import numpy as np
from bass_temporal_data import annotate
from bass_training_data import eligible, identity
from current_bass_v15_baseline import VERSION, hashes
from prepare_robust_training_stems import digest


def read_items(folder, records, plan_sha, groups, baseline):
    items = []
    for row in records:
        if row['group'] not in groups:
            continue
        path = folder / 'features' / (row['id'] + '.npz') if baseline else folder / row['id'] / 'features.npz'
        if digest(path) != row['cache_sha256'] or digest(Path(row['audio'])) != row['audio_sha256']:
            raise ValueError('Changed V15 spectral evidence or source audio')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['plan_sha256']) != plan_sha
                    or str(saved['identity']) != identity({k: v for k, v in row.items() if k != 'cache_sha256'})):
                raise ValueError('Changed V15 spectral/source identity')
            item = {**row, **{key: saved[key] for key in ('events', 'base_x', 'frames', 'eligible')}}
        n = len(item['events'])
        if (item['events'].shape != (n, 4) or n != row['retained_v15'] or item['base_x'].shape != (n, 52)
                or item['frames'].shape != (n, 40, 9) or not np.isfinite(item['base_x']).all()
                or not np.isfinite(item['events']).all() or not np.isfinite(item['frames']).all()
                or np.any((item['frames'] < 0) | (item['frames'] > 1))):
            raise ValueError('Invalid V15 temporal/acoustic evidence')
        np.testing.assert_array_equal(item['eligible'], eligible(item['events'], row['duration']))
        item.update(reference=np.asarray(row['reference'], float).reshape(-1, 3),
                    pitch_reference=np.asarray(row['pitch_reference'], float).reshape(-1, 3))
        items.append(annotate(item))
    return items


def baseline_items(folder, groups):
    plan = json.loads((folder / 'plan.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    records = manifest['items']
    if (plan['version'] != VERSION or manifest['version'] != VERSION
            or plan['baseline_hashes'] != hashes() or manifest['plan_sha256'] != digest(folder / 'plan.json')
            or plan['code_sha256'] != digest(Path(__file__).with_name('prepare_bass_v15_baseline.py'))
            or not plan['no_audio_inference'] or not manifest['all_tests_consumed_regression']
            or not manifest['no_user_audio_or_scores'] or len({r['id'] for r in records}) != 929
            or {g: sum(r['group'] == g for r in records) for g in ('train', 'validation', 'test')}
            != {'train': 531, 'validation': 146, 'test': 252}):
        raise ValueError('Changed actual V15 survivor baseline')
    partitions = {g: {r['source_group'] for r in records if r['group'] == g} for g in ('train', 'validation', 'test')}
    if any(partitions[a] & partitions[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('V15 baseline source partitions overlap')
    return read_items(folder, records, manifest['plan_sha256'], groups, True)


def piano_items(folder, groups):
    if any(g not in ('train', 'validation') for g in groups):
        raise ValueError('Real piano fitting preparation excludes test sources')
    plan = json.loads((folder / 'plan.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    audit = json.loads((folder / 'source-audit.json').read_text())
    root = Path(__file__).resolve().parents[1]
    records = manifest['items']
    expected = {'vienna-bass-v15-' + row['id'] + '-' + variant: row
                for row in plan['sources'] for variant in row['variants']}
    excluded = {row['id'] for row in audit['excluded']}
    actual = {row['id'] for row in records}
    if (plan['baseline_hashes'] != hashes() or manifest['plan_sha256'] != digest(folder / 'plan.json')
            or plan['license'] != 'CC-BY-4.0' or not manifest['no_test_audio_or_metrics']
            or not manifest['no_user_audio_or_scores'] or len(actual) != len(records)
            or actual & excluded or actual | excluded != set(expected)
            or any(not row['excluded_no_bass'] or row['plan_sha256'] != manifest['plan_sha256'] for row in audit['excluded'])
            or any(digest(root / name) != sha for name, sha in plan['code_sha256'].items())):
        raise ValueError('Changed or incomplete real-piano V15 preparation')
    for row in records:
        source = expected[row['id']]
        if row['group'] != source['group'] or row['source_group'] != source['source_group'] or row['group'] == 'test':
            raise ValueError('Changed real-piano source partition')
    return read_items(folder, records, manifest['plan_sha256'], groups, False)


def collect(baseline, piano):
    training = baseline_items(baseline, ('train',)) + piano_items(piano, ('train',))
    validation = baseline_items(baseline, ('validation',)) + piano_items(piano, ('validation',))
    if {i['source_group'] for i in training} & {i['source_group'] for i in validation}:
        raise ValueError('Expanded V15 fitting/selection source groups overlap')
    if len({i['id'] for i in training + validation}) != len(training + validation):
        raise ValueError('Duplicate expanded V15 candidates')
    return training, validation
