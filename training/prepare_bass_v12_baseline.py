"""Reuse sealed V11 acoustic intervals, applying every released V12 decision.

No audio is decoded and no user files are read. Previously first-pass recordings
are consumed regression only. Historical guards are kept unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_residual import BassResidual
from app.services.bass_verifier import BassVerifier
from bass_training_data import identity, load
from current_bass_v12_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

PARENT_SHA = 'ed0e201bf56876a7030eb8b123eb4414521bc0cab2f76530876848308fe0c015'
PARITY_SHA = '53077bf04da0b77814027566cd14bf28c4d926d1c976c4cdcae0692c8edc3715'
NATIVE_SHA = 'd5ee200a740816e3cb3b9939c71c58bc545a3730b04817ea7a7d877e5d4f269d'
FRESH_SHA = '303a925d350276565653d8045e2b5553f63dd4f5c8efe3e610ad2740da8ecc22'


def sealed_sources(run, fresh):
    for name, expected in (('plan', PARENT_SHA), ('runtime-parity', PARITY_SHA),
                           ('native-routing', NATIVE_SHA), ('original-first-pass', FRESH_SHA)):
        if digest(run / (name + '.json')) != expected:
            raise ValueError('Changed sealed V12 release evidence')
    parent = json.loads((run / 'plan.json').read_text())
    cache = Path(parent['baseline_cache_root'])
    if (digest(cache / 'manifest.json') != parent['baseline_cache_manifest_sha256']
            or digest(cache / 'plan.json') != parent['baseline_cache_plan_sha256']):
        raise ValueError('Changed sealed V11 duration cache')
    first = json.loads((run / 'original-first-pass.json').read_text())
    if (digest(fresh / 'manifest.json') != first['test_manifest_sha256']
            or not first['passes'] or not first['now_consumed_regression']):
        raise ValueError('Changed consumed original corpus')
    root = Path(__file__).resolve().parents[1]
    for baseline in (parent['baseline_hashes'], parent['baseline_hashes']['v10']):
        for name, expected in baseline['sources'].items():
            if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
                raise ValueError('Changed duration-independent historical evidence code')
    return parent, cache


def sources(run, fresh):
    parent, cache = sealed_sources(run, fresh)
    manifest = json.loads((cache / 'manifest.json').read_text())
    raw = {}
    for record in manifest['items']:
        root, group = Path(record['root']), record['group']
        key = (root, group)
        collection = parent['manifests'] if group != 'test' else parent['consumed_regression_manifests']
        if digest(root / 'manifest.json') != collection[str(root)]:
            raise ValueError('Changed sealed source manifest')
        if key not in raw:
            raw[key] = {item['id']: item for item in load(root, group)}
        item = raw[key][record['id']]
        original = next(i for i in json.loads((root / 'manifest.json').read_text())['items'] if i['id'] == item['id'])
        if identity(original) != record['identity'] or digest(Path(item['audio'])) != record['audio_sha256']:
            raise ValueError('Changed sealed note/source identity')
        path = cache / 'features' / (record['id'] + '.npz')
        if digest(path) != record['cache_sha256']:
            raise ValueError('Changed sealed acoustic interval cache')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['plan_sha256']) != manifest['plan_sha256']
                    or str(saved['identity']) != json.dumps({k: v for k, v in record.items() if k != 'cache_sha256'}, sort_keys=True)):
                raise ValueError('Changed sealed interval identity')
            evidence = {key: saved[key] for key in ('events', 'eligible')}
            evidence['base_x'] = saved['x']
            duration = float(saved['duration'])
        if sf.info(item['audio']).duration != duration:
            raise ValueError('Changed sealed audio clock')
        yield {**item, **evidence, 'duration': duration, 'source_cache_sha256': record['cache_sha256']}
    first = json.loads((fresh / 'manifest.json').read_text())
    if first['plan_sha256'] != digest(fresh / 'plan.json') or len(first['items']) != 32:
        raise ValueError('Changed consumed first-pass plan')
    for item in first['items']:
        completion = json.loads((fresh / item['id'] / 'complete.json').read_text())
        if (completion['item'] != item or completion['plan_sha256'] != first['plan_sha256']
                or completion['raw_cache_sha256'] != first['cache_sha256'][item['id']]
                or completion['post_cache_sha256'] != first['post_cache_sha256'][item['id']]
                or digest(Path(item['audio'])) != completion['audio_sha256']):
            raise ValueError('Changed consumed original audio or source identity')
        path = fresh / 'post-v11-features' / (item['id'] + '.npz')
        if item['group'] != 'test' or digest(path) != first['post_cache_sha256'][item['id']]:
            raise ValueError('Changed consumed first-pass cache')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != first['plan_sha256']:
                raise ValueError('Changed consumed first-pass identity')
            evidence = {key: saved[key] for key in ('base_x', 'events', 'eligible')}
        yield {**item, **evidence, 'reference': np.asarray(item['reference']).reshape(-1, 3),
               'pitch_reference': np.asarray(item['pitch_reference']).reshape(-1, 3),
               'duration': sf.info(item['audio']).duration, 'seconds': 29.5,
               'source_cache_sha256': first['post_cache_sha256'][item['id']]}


def prepare(run, fresh, output):
    output.mkdir(exist_ok=False)
    parent, cache = sealed_sources(run, fresh)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'parent_plan_sha256': PARENT_SHA,
            'parent_run': str(run), 'parent_cache': str(cache), 'consumed_original_root': str(fresh),
            'parent_cache_manifest_sha256': parent['baseline_cache_manifest_sha256'],
            'parent_cache_plan_sha256': parent['baseline_cache_plan_sha256'],
            'consumed_original_manifest_sha256': digest(fresh / 'manifest.json'),
            'code_sha256': digest(Path(__file__)), 'no_user_audio_or_scores': True,
            'no_audio_inference': True,
            'scope': 'All 705 sealed recordings, actual V12 retained intervals. Reuse independent base acoustic evidence only because no interval changes; recompute candidate relationships after deletion. 124 consumed regressions never fit/select.'}
    preserve(output / 'plan.json', plan)
    work = output / 'features'
    work.mkdir()
    baseline, residual, records, seen = BassVerifier(), BassResidual(), [], set()
    groups = {name: set() for name in ('train', 'validation', 'test')}
    for item in sources(run, fresh):
        if item['id'] in seen:
            raise ValueError('Duplicated V12 cache identity')
        seen.add(item['id'])
        groups[item['group']].add(item['source_group'])
        result = decisions(item, item['duration'], baseline, residual)
        record = {key: item[key] for key in ('id', 'group', 'source_group', 'corpus', 'audio', 'seconds', 'source_cache_sha256')}
        record['seconds'] = float(record['seconds'])
        record.update(reference=item['reference'].tolist(), pitch_reference=item['pitch_reference'].tolist(),
                      duration=item['duration'], retained_v11=len(item['events']), retained_v12=len(result['events']),
                      residual_rejections=result['residual_rejections'], audio_sha256=digest(Path(item['audio'])))
        path = work / (item['id'] + '.npz')
        np.savez_compressed(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                            identity=identity(record), plan_sha256=digest(output / 'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
    if (len(records) != 705 or any(groups[a] & groups[b] for a, b in
            (('train', 'validation'), ('train', 'test'), ('validation', 'test')))):
        raise ValueError('V12 cache count or source partition differs from sealed release')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_user_audio_or_scores': True, 'all_tests_consumed_regression': True})
    print(json.dumps({'recordings': len(records), 'v12_rejections': sum(r['residual_rejections'] for r in records),
                     'groups': {g: sum(r['group'] == g for r in records) for g in groups},
                     'no_audio_inference': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.run, args.fresh, args.output)
