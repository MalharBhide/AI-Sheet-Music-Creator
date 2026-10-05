"""Read sealed current-bass examples; test partitions require explicit callers."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v16_baseline import VERSION, hashes
from prepare_bass_v16_baseline import COUNTS
from prepare_robust_training_stems import digest


def load(folder, groups=('train', 'validation')):
    plan = json.loads((folder/'plan.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    if (plan['version'] != VERSION or manifest['version'] != VERSION
            or manifest['plan_sha256'] != digest(folder/'plan.json') or plan['baseline_hashes'] != hashes()
            or plan['producer_sha256'] != digest(Path(__file__).with_name('prepare_bass_v16_baseline.py'))
            or not plan['no_audio_inference'] or not plan['no_user_audio_or_scores']
            or not manifest['all_tests_consumed_regression'] or not manifest['no_user_audio_or_scores']):
        raise ValueError('Changed V16 fitting baseline')
    records = manifest['items']
    if (len({r['id'] for r in records}) != 1029
            or {g:sum(r['group'] == g for r in records) for g in COUNTS} != COUNTS
            or any(g not in COUNTS for g in groups)):
        raise ValueError('Invalid V16 source coverage or partition')
    partitions = {g:{r['source_group'] for r in records if r['group'] == g} for g in COUNTS}
    if any(partitions[a]&partitions[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('V16 source-group leakage')
    result = []
    for record in records:
        if record['group'] not in groups:
            continue
        path = folder/'features'/(record['id']+'.npz')
        if digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
            raise ValueError('Changed V16 spectral/acoustic evidence')
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved['plan_sha256']) != manifest['plan_sha256']
                    or str(saved['identity']) != identity({k:v for k,v in record.items() if k != 'cache_sha256'})):
                raise ValueError('Changed V16 cache interval identity')
            item = {**record, **{key:saved[key] for key in ('events','base_x','frames','eligible')}}
        n = len(item['events'])
        if (n != record['retained_v16'] or item['events'].shape != (n,4)
                or item['base_x'].shape != (n,52) or item['frames'].shape != (n,40,9)
                or any(not np.isfinite(item[key]).all() for key in ('events','base_x','frames'))
                or np.any((item['frames']<0)|(item['frames']>1))):
            raise ValueError('Invalid V16 interval-aligned evidence')
        np.testing.assert_array_equal(item['eligible'], eligible(item['events'],item['duration']))
        item.update(x=features(item['events'],item['base_x']),
                    reference=np.asarray(record['reference'],float).reshape(-1,3),
                    pitch_reference=np.asarray(record['pitch_reference'],float).reshape(-1,3))
        result.append(item)
    return result
