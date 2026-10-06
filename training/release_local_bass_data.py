"""Load release-window fitting evidence with sealed attack-window and parent identities."""

import json
from pathlib import Path

import numpy as np
from attack_local_bass_data import load as attack_items
from attack_local_features import CHANNELS, HOP, RATE, STEPS
from bass_training_data import identity
from bass_v16_data import load as parent_items
from current_bass_v16_baseline import hashes
from prepare_release_local_bass import contracts
from prepare_robust_training_stems import digest
from release_local_features import VERSION


def load(folder, groups=('train','validation')):
    if any(g not in ('train','validation') for g in groups):
        raise ValueError('Release-local fitting cannot read regression partitions')
    plan=json.loads((folder/'plan.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    if (plan['version']!=VERSION or manifest['version']!=VERSION or plan['baseline_hashes']!=hashes()
            or plan['code_sha256']!=contracts() or manifest['plan_sha256']!=digest(folder/'plan.json')
            or not plan['no_test_audio_or_metrics'] or not manifest['no_test_audio_or_metrics']
            or not plan['no_user_audio_or_scores'] or not manifest['no_user_audio_or_scores']
            or (plan['rate'],plan['hop'],plan['steps'],plan['channels'])!=(RATE,HOP,STEPS,CHANNELS)):
        raise ValueError('Changed release-local preparation contract')
    parent=Path(plan['parent_root'])
    if digest(parent/'manifest.json')!=plan['parent_manifest_sha256'] or digest(parent/'plan.json')!=plan['parent_plan_sha256']:
        raise ValueError('Changed parent V16 baseline')
    records=manifest['items']
    if (len(records)!=733 or len({r['id'] for r in records})!=733
            or {g:sum(r['group']==g for r in records) for g in ('train','validation')}!={'train':573,'validation':160}):
        raise ValueError('Incomplete release-local fitting coverage')
    parents={i['id']:i for i in parent_items(parent,groups)}
    result=[]
    for record in records:
        if record['group'] not in groups:
            continue
        item=parents.pop(record['id'])
        if (record['parent_cache_sha256']!=item['cache_sha256'] or record['group']!=item['group']
                or record['source_group']!=item['source_group'] or record['corpus']!=item['corpus']
                or record['audio_sha256']!=item['audio_sha256']):
            raise ValueError('Changed release-local source/interval identity')
        path=folder/'features'/(item['id']+'.npz')
        if digest(path)!=record['release_cache_sha256']:
            raise ValueError('Changed release-local feature cache')
        with np.load(path,allow_pickle=False) as saved:
            if (str(saved['plan_sha256'])!=manifest['plan_sha256']
                    or str(saved['identity'])!=identity({k:v for k,v in record.items() if k!='release_cache_sha256'})):
                raise ValueError('Changed release-local cache identity')
            np.testing.assert_array_equal(saved['events'],item['events'])
            frames=saved['release_frames']
        if (frames.shape!=(len(item['events']),STEPS,CHANNELS) or not np.isfinite(frames).all()
                or np.any((frames<0)|(frames>1))):
            raise ValueError('Invalid release-local fine envelopes')
        result.append({**item,'release_frames':frames})
    if parents:
        raise ValueError('Missing release-local source records')
    attack={i['id']:i for i in attack_items(parent.parent/'attack-local-bass-v1',groups)}
    if set(attack)!={i['id'] for i in result}:
        raise ValueError('Attack/release fitting sources differ')
    for item in result:
        twin=attack[item['id']]
        np.testing.assert_array_equal(item['events'],twin['events'])
        item['attack_frames']=twin['attack_frames']
    return result
