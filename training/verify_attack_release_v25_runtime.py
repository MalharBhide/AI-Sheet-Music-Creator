"""Portable probability, decision and source-note parity on every sealed input."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from app.services.bass_harmonic import features
from attack_release_model import probabilities
from bass_joint_timing import actions
from bass_joint_timing_v18 import margin
from bass_v16_data import load as parent_items
from evaluate_attack_release_v25 import (
    attack_evidence,
    previous_fresh_observations,
    release_evidence,
)
from pitch_consensus_early import transform
from prepare_robust_training_stems import digest, preserve
from release_attack_release_v25 import sealed
from release_local_bass_data import load


def runtime(prototype):
    if prototype is not None:
        for name in ('attack_local_evidence','release_local_evidence','attack_local_network','bass_attack_declutter'):
            full='app.services.'+name
            spec=importlib.util.spec_from_file_location(full,prototype/(name+'.py'))
            module=importlib.util.module_from_spec(spec)
            sys.modules[full]=module
            spec.loader.exec_module(module)
    from app.services import bass_attack_declutter
    return bass_attack_declutter


def all_items(run,fresh,plan,winner):
    fitting=load(Path(plan['baseline_root']))
    parent=Path(json.loads((Path(plan['baseline_root'])/'plan.json').read_text())['parent_root'])
    consumed=parent_items(parent,('test',))
    consumed=attack_evidence(run,plan,winner,consumed)
    for previous in plan['previous_fresh']:
        consumed.extend(previous_fresh_observations(previous))
    consumed=release_evidence(run,plan,winner,consumed)
    first=previous_fresh_observations({'root':str(fresh),'manifest_sha256':digest(fresh/'manifest.json'),
                                      'plan_sha256':digest(fresh/'plan.json')})
    for item in first:
        with np.load(fresh/item['id']/'features.npz',allow_pickle=False) as saved:
            item['release_frames']=saved['release_frames']
    if (len(fitting)!=733 or len(consumed)!=360 or len(first)!=32
            or len({i['id'] for i in fitting+consumed+first})!=1125):
        raise ValueError('Incomplete runtime parity corpus')
    return fitting+consumed+first


def verify(run,fresh,prototype):
    report=run/'runtime-parity.json'
    if report.exists():
        raise ValueError('Preserve existing runtime parity result')
    plan,winner,network,normalizer=sealed(run,fresh)
    export=json.loads((run/'portable/export.json').read_text())
    asset=run/'portable/bass-attack-declutter-v1.npz'
    service=runtime(prototype)
    if (export['asset_sha256']!=digest(asset) or export['asset_sha256']!=service.EXPECTED_SHA
            or export['checkpoint_sha256']!=winner['checkpoint_sha256']):
        raise ValueError('Changed portable candidate or runtime checksum contract')
    items=all_items(run,fresh,plan,winner)
    with patch.object(service,'ASSET',asset):
        verifier=service.BassAttackDeclutter()
    maximum_error=0.
    events,removed=0,0
    for item in items:
        x=features(item['events'],item['base_x'])
        np.testing.assert_array_equal(x,item['x'],err_msg=item['id'])
        torch.set_num_threads(2)
        expected=probabilities(network,item['frames'],item['attack_frames'],item['release_frames'],x,normalizer)
        guardian=verifier.guardian.probability(item['base_x'])
        for threads in (1,2):
            torch.set_num_threads(threads)
            actual=verifier.model.probability(item['frames'],item['attack_frames'],item['release_frames'],x)
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6,err_msg=item['id'])
            maximum_error=max(maximum_error,float(np.max(np.abs(actual-expected),initial=0.)))
            for stricter in (False,True):
                p=np.zeros((len(expected),4),np.float32)
                p[:,1],p[:,3]=expected[:,0],expected[:,1]
                threshold=margin(winner['threshold']) if stricter else winner['threshold']
                chosen=actions(item,p,np.where(guardian<winner['guardian_threshold'],0.,1.),1.01,threshold)
                after,indices=transform(item['events'],chosen,item['duration'])
                keep=verifier.model.keep(item['events'],actual,guardian,item['duration'],stricter)
                np.testing.assert_array_equal(np.flatnonzero(keep),indices,err_msg=item['id'])
                np.testing.assert_array_equal(after,item['events'][keep],err_msg=item['id'])
        notes=[SimpleNamespace(start=float(s),end=float(e),pitch=int(p),velocity=int(v)) for s,e,p,v in item['events']]
        original=[vars(n).copy() for n in notes]
        for parts,order in (([notes],list(range(len(notes)))),
            ([notes[::2],notes[1::2]],list(range(0,len(notes),2))+list(range(1,len(notes),2)))):
            ordered={key:item[key][order] for key in ('base_x','frames','attack_frames','release_frames')}
            ox=features(item['events'][order],ordered['base_x'])
            op=probabilities(network,ordered['frames'],ordered['attack_frames'],ordered['release_frames'],ox,normalizer)
            og=verifier.guardian.probability(ordered['base_x'])
            keep=verifier.model.keep(item['events'][order],op,og,item['duration'])
            midi=SimpleNamespace(instruments=[SimpleNamespace(notes=part.copy()) for part in parts])
            with patch.object(service.sf,'read',return_value=(np.zeros(round(item['duration']*22050),np.float32),22050)), \
                 patch.object(service,'note_features',return_value=ordered['base_x']), \
                 patch.object(service,'observe',return_value={'cqt':None}), \
                 patch.object(service,'coarse_sequences',return_value=ordered['frames']), \
                 patch.object(service,'attack_sequences',return_value=ordered['attack_frames']), \
                 patch.object(service,'release_sequences',return_value=ordered['release_frames']):
                count=verifier.filter('sealed-dataset-evidence.wav',{},midi)
            assert torch.get_num_threads()==2,item['id']
            assert count==int((~keep).sum()),item['id']
            kept_ids={id(notes[i]) for i,k in zip(order,keep,strict=True) if k}
            for part,prior in zip(midi.instruments,parts,strict=True):
                assert [id(n) for n in part.notes]==[id(n) for n in prior if id(n) in kept_ids],item['id']
            assert original==[vars(n) for n in notes],item['id']
        events+=len(notes)
        removed+=int((~verifier.model.keep(item['events'],expected,guardian,item['duration'])).sum())
    sealed(run,fresh)
    paths={name:Path(sys.modules['app.services.'+name].__file__) for name in
           ('attack_local_evidence','release_local_evidence','attack_local_network','bass_attack_declutter')}
    result={'passes':True,'recordings':len(items),'partitions':{'train':573,'validation':160,'consumed':360,'first_pass':32},
            'input_events':events,'removed_events':removed,'maximum_probability_error':maximum_error,'tolerance':1e-6,
            'thread_counts':[1,2],'both_confidence_decisions_identical':True,
            'retained_pitch_clock_velocity_object_part_identity_preserved':True,
            'global_thread_setting_unchanged_by_runtime':True,'asset_sha256':digest(asset),
            'checkpoint_sha256':winner['checkpoint_sha256'],'batch_selection_sha256':digest(run/'batch-selection.json'),
            'runtime_source_sha256':{name:digest(path) for name,path in paths.items()},'script_sha256':digest(Path(__file__)),
            'no_user_audio_or_scores':True,'physical_feature_and_native_pipeline_checks_still_required':True}
    preserve(report,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('fresh',type=Path)
    parser.add_argument('--prototype',type=Path)
    args=parser.parse_args()
    verify(args.run,args.fresh,args.prototype)
