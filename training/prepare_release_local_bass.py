"""Cache observed note endings for training/validation; no inference or regression audio."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from attack_local_features import CHANNELS, HOP, RATE, STEPS, observe
from bass_training_data import identity
from bass_v16_data import load
from current_bass_v16_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from release_local_features import VERSION, release_sequences


def contracts():
    root=Path(__file__).resolve().parents[1]
    names=('training/prepare_release_local_bass.py','training/attack_local_features.py','training/release_local_features.py','training/bass_v16_data.py',
           'training/current_bass_v16_baseline.py','backend/app/services/temporal_note_model.py')
    return {name:digest(root/name) for name in names}


def prepare(parent, output):
    items=load(parent)
    if (len(items)!=733
            or {g:sum(i['group']==g for i in items) for g in ('train','validation')}!={'train':573,'validation':160}
            or len({i['id'] for i in items})!=733
            or {i['source_group'] for i in items if i['group']=='train'}
               & {i['source_group'] for i in items if i['group']=='validation'}):
        raise ValueError('Invalid release-local fitting partition')
    plan={'version':VERSION,'baseline_hashes':hashes(),'code_sha256':contracts(),'parent_root':str(parent),
          'parent_manifest_sha256':digest(parent/'manifest.json'),'parent_plan_sha256':digest(parent/'plan.json'),
          'groups':{'train':573,'validation':160},'rate':RATE,'hop':HOP,'steps':STEPS,'channels':CHANNELS,
          'feature_scope':'61 observed positions centered on predicted ending; 9 pitch-relative CQT channels, short/long STFT fundamental and 2nd–4th harmonic bands, short-window flux. No labels or fitted probabilities.',
          'no_test_audio_or_metrics':True,'no_user_audio_or_scores':True}
    if output.exists():
        if json.loads((output/'plan.json').read_text())!=plan or (output/'manifest.json').exists():
            raise ValueError('Preserve completed or changed release-local preparation')
    else:
        output.mkdir()
        preserve(output/'plan.json',plan)
    folder=output/'features'
    folder.mkdir(exist_ok=True)
    records=[]
    for item in items:
        record={k:item[k] for k in ('id','group','source_group','corpus','audio_sha256','cache_sha256')}
        record['parent_cache_sha256']=record.pop('cache_sha256')
        path=folder/(item['id']+'.npz')
        completion=folder/(item['id']+'.json')
        if completion.exists():
            saved=json.loads(completion.read_text())
            if (saved['record']!={**record,'release_cache_sha256':digest(path)}
                    or saved['plan_sha256']!=digest(output/'plan.json')):
                raise ValueError('Changed completed release-local record')
            record=saved['record']
        else:
            samples,rate=sf.read(item['audio'],dtype='float32')
            if digest(Path(item['audio']))!=item['audio_sha256'] or rate!=RATE or samples.ndim!=1 or len(samples)/RATE!=item['duration']:
                raise ValueError('Changed fitting waveform clock')
            frames=release_sequences(observe(samples,rate),item['events'])
            np.savez_compressed(path,release_frames=frames,events=item['events'],identity=identity(record),
                                plan_sha256=digest(output/'plan.json'))
            record['release_cache_sha256']=digest(path)
            preserve(completion,{'record':record,'plan_sha256':digest(output/'plan.json')})
        records.append(record)
        if len(records)%20==0:
            print(json.dumps({'cached':len(records),'total':len(items),'id':item['id']}),flush=True)
    if plan['code_sha256']!=contracts() or plan['baseline_hashes']!=hashes():
        raise ValueError('Release-local source/baseline changed during preparation')
    preserve(output/'manifest.json',{'version':VERSION,'plan_sha256':digest(output/'plan.json'),'items':records,
             'no_test_audio_or_metrics':True,'no_user_audio_or_scores':True})
    print(json.dumps({'complete':True,'recordings':len(records),'manifest_sha256':digest(output/'manifest.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    prepare(args.parent,args.output)
