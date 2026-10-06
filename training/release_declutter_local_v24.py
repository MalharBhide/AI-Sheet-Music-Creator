"""Release only a frozen candidate with complete positive regression gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from bass_joint_timing_v18 import margin
from evaluate_declutter_local_v24 import frozen
from fresh_declutter_local_v24 import require_regression
from prepare_robust_training_stems import digest, preserve
from train_declutter_local_v24 import VERSION


def sealed(run,fresh):
    plan,winner,network,normalizer=frozen(run)
    regression=require_regression(run,winner)
    result=json.loads((run/'original-first-pass.json').read_text())
    manifest=json.loads((fresh/'manifest.json').read_text())
    prepared=json.loads((fresh/'plan.json').read_text())
    root=Path(__file__).resolve().parents[1]
    if (not result['passes'] or not result['first_pass_complete'] or not result['no_retuning']
            or not result['now_consumed_regression'] or result['recordings']!=32
            or result['checkpoint_sha256']!=winner['checkpoint_sha256']
            or result['batch_selection_sha256']!=digest(run/'batch-selection.json')
            or result['test_manifest_sha256']!=digest(fresh/'manifest.json')
            or manifest['plan_sha256']!=digest(fresh/'plan.json') or len(manifest['items'])!=32
            or not manifest['now_consumed_regression'] or not manifest['no_fitting_selection_user_audio_or_scores']
            or len(plan['fresh_seeds'])!=32 or len(set(plan['fresh_seeds']))!=32
            or len({i['id'] for i in manifest['items']})!=32
            or {i['source_group'] for i in manifest['items']}!={'original-bass-seed-'+str(seed) for seed in plan['fresh_seeds']}
            or any(i['group']!='test' for i in manifest['items'])
            or prepared['seeds']!=plan['fresh_seeds']
            or prepared['baseline_hashes']!=plan['baseline_hashes']
            or prepared['batch_selection_sha256']!=digest(run/'batch-selection.json')
            or 'training/fresh_declutter_local_v24.py' not in prepared['code_sha256']
            or prepared['checkpoint_sha256']!=winner['checkpoint_sha256']
            or prepared['consumed_regression_sha256']!=regression
            or not prepared['no_fitting_selection_user_audio_or_scores']
            or any(digest(root/name)!=sha for name,sha in prepared['code_sha256'].items())):
        raise ValueError('Missing or changed complete first-pass release gate')
    for name,t in (('raw',winner['threshold']), ('margin',margin(winner['threshold']))):
        row=result[name]
        if (not row['passes'] or len(row['per_recording'])!=32
                or any(not i['passes'] for i in row['per_recording'])
                or row['false_notes_removed']<=0 or row['retimed_notes']
                or abs(row['onset_error_reduction_seconds'])>1e-9
                or (row['remove_threshold'],row['guardian_threshold'])!=(t,winner['guardian_threshold'])):
            raise ValueError('Changed first-pass decision configuration or preservation gates')
    return plan,winner,network,normalizer


def export(run,fresh,output):
    plan,winner,network,normalizer=sealed(run,fresh)
    output.mkdir(exist_ok=False)
    path=output/'bass-attack-declutter-v1.npz'
    mean,scale=normalizer
    fields={'version':VERSION,'format_version':1,'width':winner['width'],
            'remove_threshold':winner['threshold'],'guardian_threshold':winner['guardian_threshold'],'mean':np.asarray(mean,np.float32),'scale':np.asarray(scale,np.float32)}
    fields.update({'weight_'+key:value.numpy() for key,value in network.state_dict().items()})
    np.savez_compressed(path,**fields)
    sealed(run,fresh)
    report={'passes':True,'version':VERSION,'checkpoint_sha256':winner['checkpoint_sha256'],
            'batch_selection_sha256':digest(run/'batch-selection.json'),'asset_sha256':digest(path),
            'plan_sha256':digest(run/'plan.json'),'first_pass_sha256':digest(run/'original-first-pass.json'),
            'consumed_regression_sha256':digest(run/'consumed-regression.json'),
            'fresh_manifest_sha256':digest(fresh/'manifest.json'),'source_sha256':digest(Path(__file__)),
            'frozen_selection':winner,'no_user_audio_or_scores':True,'runtime_parity_still_required':True}
    preserve(output/'export.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('fresh',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    export(args.run,args.fresh,args.output)
