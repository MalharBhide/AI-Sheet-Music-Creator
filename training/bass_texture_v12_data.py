"""Fixed partitions of sealed V12 notes plus original acoustic texture variants."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_verifier import BassVerifier
from bass_residual_coverage_data import protect
from bass_residual_v11_data import annotate
from bass_residual_v12_data import load_cached as retained
from bass_salience_v12_data import ACOUSTIC_NAMES, NAMES, acoustic_view, features
from bass_training_data import identity
from current_bass_v12_baseline import hashes
from prepare_robust_training_stems import digest

VERSION = 'bass-v12-acoustic-texture-v3'
__all__ = ['VERSION', 'NAMES', 'ACOUSTIC_NAMES', 'acoustic_view', 'features', 'collect', 'load_cached', 'catalog']


def catalog(directory):
    plan = json.loads((directory / 'plan.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    parent, consumed = Path(plan['parent_cache_root']), Path(plan['consumed_first_pass_root'])
    root = Path(__file__).resolve().parents[1]
    if (plan['baseline_hashes'] != hashes() or not manifest['no_test_inference']
            or not manifest['no_user_audio_or_scores'] or manifest['plan_sha256'] != digest(directory / 'plan.json')
            or digest(parent / 'manifest.json') != plan['parent_cache_manifest_sha256']
            or digest(parent / 'plan.json') != plan['parent_cache_plan_sha256']
            or digest(consumed / 'manifest.json') != plan['consumed_first_pass_manifest_sha256']
            or any(digest(root / name) != expected for name, expected in plan['code_sha256'].items())):
        raise ValueError('Changed texture baseline/source contract')
    originals = json.loads((parent / 'manifest.json').read_text())
    first = json.loads((consumed / 'manifest.json').read_text())
    if not first['now_consumed_regression'] or not first['no_fitting_selection_user_audio_or_scores']:
        raise ValueError('Prior first-pass data cannot become fitting or selection')
    rows = originals['items'] + first['items'] + manifest['items']
    expected = {'train': 531, 'validation': 146, 'test': 156}
    if (len({r['id'] for r in rows}) != 833
            or {g: sum(r['group'] == g for r in rows) for g in expected} != expected
            or any(i['group'] != 'test' for i in first['items'])):
        raise ValueError('Incomplete or duplicate texture partitions')
    groups = {g: {r['source_group'] for r in rows if r['group'] == g} for g in expected}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Texture source partitions leak')
    return plan, manifest, first, rows


def load_cached(directory, groups):
    plan, manifest, first, _ = catalog(directory)
    parent, consumed = Path(plan['parent_cache_root']), Path(plan['consumed_first_pass_root'])
    result = [{**item, 'x': features(item['events'], item['base_x'])} for item in retained(parent, groups)]
    baseline = BassVerifier()
    for collection, folder, fresh in ((manifest, directory, False), (first, consumed, True)):
        for record in collection['items']:
            if record['group'] not in groups:
                continue
            path = folder / record['id'] / 'features.npz'
            if digest(path) != record['cache_sha256']:
                raise ValueError('Changed texture or consumed feature cache')
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['plan_sha256']) != collection['plan_sha256']:
                    raise ValueError('Changed texture or consumed feature identity')
                if not fresh and str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'}):
                    raise ValueError('Changed texture note/source identity')
                item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible')}}
                if fresh:
                    np.testing.assert_array_equal(saved['x'], features(item['events'], item['base_x']))
            if digest(Path(record['audio'])) != record['audio_sha256']:
                raise ValueError('Changed texture or consumed audio identity')
            item.update(reference=np.asarray(record['reference']).reshape(-1, 3),
                        pitch_reference=np.asarray(record['pitch_reference']).reshape(-1, 3))
            p, g = [m.probability(item['base_x'][:, :m.feature_count]) for m in baseline.models]
            item.update(context=p, guardian=g, x=item['base_x'])
            labeled = protect(annotate(item))
            result.append({**labeled, 'x': features(item['events'], item['base_x'])})
    return result


def collect(directory):
    training, validation = [load_cached(directory, (group,)) for group in ('train', 'validation')]
    if len(training) != 531 or len(validation) != 146:
        raise ValueError('Texture fitting cannot consume an incomplete fixed partition')
    return training, validation
