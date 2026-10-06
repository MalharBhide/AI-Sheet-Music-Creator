"""Physical feature and filter parity on four consumed project-authored MP3 fixtures."""

import argparse
import contextlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from app.services.bass_harmonic import features
from app.services.piano_transcription import _GeneralEngine
from attack_release_model import probabilities
from prepare_robust_training_stems import digest, preserve
from release_attack_release_v25 import sealed
from verify_attack_release_v25_runtime import runtime


def verify(run,fresh,prototype):
    report=run/'physical-runtime-parity.json'
    if report.exists():
        raise ValueError('Preserve completed physical parity evidence')
    _,winner,network,normalizer=sealed(run,fresh)
    portable=json.loads((run/'portable/export.json').read_text())
    artifact=run/'portable/bass-attack-declutter-v1.npz'
    service=runtime(prototype)
    if digest(artifact)!=portable['asset_sha256'] or service.EXPECTED_SHA!=portable['asset_sha256']:
        raise ValueError('Changed frozen portable candidate')
    with patch.object(service,'ASSET',artifact):
        verifier=service.BassAttackDeclutter()
    manifest=json.loads((fresh/'manifest.json').read_text())
    # Both existing fixture families; these were consumed by the frozen
    # first-pass check. No new seed, checkpoint selection or model tuning.
    records=[manifest['items'][i] for i in (0,1,16,17)]
    rows=[]
    for record in records:
        path=Path(record['audio'])
        assert digest(path)==record['audio_sha256']
        cache=fresh/record['id']/'features.npz'
        assert digest(cache)==record['cache_sha256']
        with np.load(cache,allow_pickle=False) as saved:
            expected={k:saved[k] for k in ('events','base_x','x','frames','attack_frames','release_frames')}
        engine,evidence=_GeneralEngine('balanced'),{}
        native=engine.predict_function

        def capture(*args,**kwargs):
            result=native(*args,**kwargs)
            evidence['acoustic']=result[0]
            return result

        engine.predict_function=capture
        torch.set_num_threads(2)
        with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet):
            midi=engine.predict(path,'bass',90.)
        notes=[n for part in midi.instruments for n in part.notes]
        events=np.asarray([[n.start,n.end,n.pitch,n.velocity] for n in notes],float).reshape(-1,4)
        np.testing.assert_array_equal(events,expected['events'],err_msg=record['id'])
        samples,rate=sf.read(path,dtype='float32')
        base=service.note_features(samples,rate,evidence['acoustic'],notes,include_context=True)
        observed=service.observe(samples,rate)
        physical={'base_x':base,'x':features(events,base),'frames':service.coarse_sequences(observed['cqt'],events),
                  'attack_frames':service.attack_sequences(observed,events),
                  'release_frames':service.release_sequences(observed,events)}
        maximum={}
        for key,actual in physical.items():
            np.testing.assert_allclose(actual,expected[key],rtol=1e-6,atol=1e-6,err_msg=record['id']+' '+key)
            maximum[key]=float(np.max(np.abs(actual-expected[key]),initial=0.))
        p=probabilities(network,expected['frames'],expected['attack_frames'],expected['release_frames'],expected['x'],normalizer)
        g=verifier.guardian.probability(expected['base_x'])
        keep=verifier.model.keep(events,p,g,len(samples)/rate)
        before=[vars(n).copy() for n in notes]
        parts=midi.instruments
        part_ids=[id(part) for part in parts]
        original_parts=[part.notes.copy() for part in parts]
        removed=verifier.filter(path,evidence['acoustic'],midi)
        assert removed==int((~keep).sum())
        ids={id(n) for n,k in zip(notes,keep,strict=True) if k}
        assert midi.instruments is parts and [id(part) for part in parts]==part_ids
        for part,prior in zip(parts,original_parts,strict=True):
            assert [id(n) for n in part.notes]==[id(n) for n in prior if id(n) in ids]
        assert before==[vars(n) for n in notes]
        rows.append({'id':record['id'],'source_group':record['source_group'],'events':len(events),'removed':removed,
                     'source_audio_sha256':record['audio_sha256'],'cache_sha256':record['cache_sha256'],
                     'maximum_feature_error':maximum,'all_retained_fields_objects_parts_preserved':True})
        print(json.dumps(rows[-1]),flush=True)
    sealed(run,fresh)
    result={'passes':True,'recordings':4,'per_recording':rows,'asset_sha256':digest(artifact),
            'checkpoint_sha256':winner['checkpoint_sha256'],'first_pass_manifest_sha256':digest(fresh/'manifest.json'),
            'source_sha256':digest(Path(__file__)),'tolerance':1e-6,'no_user_audio_or_scores':True,
            'scope':'Four consumed project-authored MP3 fixtures, two per family. Native unchanged V16 baseline, physical context/coarse/attack/ending features, actual portable filter and exact source note/part retention. No new test or accuracy claim.'}
    preserve(report,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_recording'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('fresh',type=Path)
    parser.add_argument('--prototype',type=Path)
    args=parser.parse_args()
    verify(args.run,args.fresh,args.prototype)
