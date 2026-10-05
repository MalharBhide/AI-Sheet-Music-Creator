"""Explain failed final-stage validation only, without selecting a new winner."""

import argparse
import importlib
import json
from pathlib import Path

from app.services.bass_harmonic import BassHarmonic
from bass_joint_timing import probabilities
from bass_v16_data import load
from prepare_robust_training_stems import digest, preserve
from score_bass_joint_timing import evaluate


def diagnose(run, batch):
    module = importlib.import_module('train_bass_joint_timing_'+batch)
    plan = json.loads((run/'plan.json').read_text())
    selection = json.loads((run/'batch-selection.json').read_text())
    if (selection['selected'] or selection['winner'] is not None or plan['code_sha256']!=module.contracts()
            or selection['plan_sha256']!=digest(run/'plan.json') or plan['test_used_for_selection']):
        raise ValueError('Expected a completed unchanged rejected validation batch')
    items = load(Path(plan['baseline_root']),('validation',))
    if len(items)!=160:
        raise ValueError('Incomplete validation diagnostic partition')
    guardian = BassHarmonic().models[1]
    guards = [guardian.probability(i['base_x']) for i in items]
    output=[]
    for profile in plan['profiles']:
        path=run/profile['name']/f"epoch-{max(plan['stages']):03d}.pt"
        network,normalizer = module.checkpoint(path)
        p = [probabilities(network,i['frames'],i['x'],normalizer) for i in items]
        # This is an explanation of a rejected final stage at a fixed visible
        # confidence, never a fallback release or a new threshold selection.
        result=evaluate(items,p,guards,.95,.99)
        failures=[]
        for row in result['per_recording']:
            if row['passes']:
                continue
            clock=row['timing']
            before,after=clock['before_onset_release_error_sum'],clock['after_onset_release_error_sum']
            reasons={'lost_attacks':row['matches']['attack']['lost'],
                     'lost_holds':row['matches']['hold']['lost'],
                     'lost_pitch_seconds':row['coverage']['lost_reference_seconds'],
                     'onset_error_increase_seconds':max(0.,after[0]-before[0]),
                     'release_error_increase_seconds':max(0.,after[1]-before[1]),
                     'repeated_spacing_error_increase_seconds':max(0.,clock['after_repeated_spacing_error_sum']-clock['before_repeated_spacing_error_sum']),
                     'extra_false_positives':max(0,row['candidate']['false_positives']-row['baseline']['false_positives'])}
            failures.append({'id':row['id'],'reasons':reasons})
        output.append({'profile':profile['name'],'checkpoint_sha256':digest(path),
                       'timing_threshold':.95,'remove_threshold':.99,
                       'passes':result['passes'],'onset_error_reduction_seconds':result['onset_error_reduction_seconds'],
                       'retimed_notes':result['retimed_notes'],'false_notes_removed':result['false_notes_removed'],
                       'failures':failures,'aggregate':result['aggregate']})
    preserve(run/'validation-diagnostics.json',{'validation_only':True,'recordings':160,
             'test_audio_or_metrics_used':False,'new_winner_selected':False,'deployed':False,
             'plan_sha256':digest(run/'plan.json'),'producer_sha256':digest(Path(__file__)),
             'scope':'Explain rejected final-stage validation. Aggregate gains do not waive any per-recording gate.',
             'profiles':output})
    print(json.dumps({'batch':batch,'profiles':[{k:v for k,v in row.items() if k!='aggregate'} for row in output]},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('batch',choices=('v17','v18'))
    args=parser.parse_args()
    diagnose(args.run,args.batch)
