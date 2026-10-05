"""Evaluate one frozen winner on all 296 consumed regression recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import BassHarmonic
from attack_local_features import observe, sequences
from attack_local_model import VERSION, probabilities
from bass_joint_timing_v18 import margin as margin_threshold
from bass_v16_data import load
from current_bass_v16_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from score_bass_joint_timing import evaluate
from train_attack_local_bass_v19 import checkpoint, contracts


def frozen(run):
    plan = json.loads((run/'plan.json').read_text())
    selection = json.loads((run/'batch-selection.json').read_text())
    if (plan['version']!=VERSION or plan['code_sha256']!=contracts() or plan['baseline_hashes']!=hashes()
            or selection['plan_sha256']!=digest(run/'plan.json') or selection['test_used_for_selection']
            or plan['test_used_for_selection'] or not plan['no_user_audio_or_scores']
            or not selection['no_user_audio_or_scores'] or not selection['selected']):
        raise ValueError('No safe frozen validation winner or changed fitting contract')
    winner = selection['winner']
    folder = run/winner['profile']
    standalone = json.loads((folder/'selection.json').read_text())
    validation = json.loads((folder/'validation.json').read_text())
    baseline = Path(plan['baseline_root'])
    if (winner!=standalone or winner['plan_sha256']!=selection['plan_sha256'] or winner['test_used_for_selection']
            or digest(run/winner['checkpoint'])!=winner['checkpoint_sha256']
            or digest(folder/'validation.json')!=winner['validation_sha256']
            or digest(baseline/'manifest.json')!=plan['baseline_manifest_sha256']
            or digest(baseline/'plan.json')!=plan['baseline_plan_sha256']
            or any(not validation[k]['passes'] or validation[k]['onset_error_reduction_seconds']<=1e-6
                   for k in ('raw','margin'))
            or validation['raw']['timing_threshold']!=winner['timing_threshold']
            or validation['raw']['remove_threshold']!=winner['remove_threshold']):
        raise ValueError('Changed selected timing checkpoint, evidence or thresholds')
    network,normalizer = checkpoint(run/winner['checkpoint'])
    return plan,winner,network,normalizer


def attack_evidence(run,plan,winner,items):
    folder=run/'consumed-attack-features'
    evidence_plan={'checkpoint_sha256':winner['checkpoint_sha256'],
                   'batch_selection_sha256':digest(run/'batch-selection.json'),
                   'baseline_hashes':plan['baseline_hashes'],'code_sha256':plan['code_sha256'],
                   'no_fitting_selection_or_user_audio':True,'recordings':296}
    if folder.exists():
        if json.loads((folder/'plan.json').read_text())!=evidence_plan:
            raise ValueError('Changed consumed attack preparation')
    else:
        folder.mkdir()
        preserve(folder/'plan.json',evidence_plan)
    result,records=[],[]
    for item in items:
        path=folder/(item['id']+'.npz')
        record={'id':item['id'],'source_group':item['source_group'],'audio_sha256':item['audio_sha256'],
                'parent_cache_sha256':item['cache_sha256'],'plan_sha256':digest(folder/'plan.json')}
        record_path=folder/(item['id']+'.json')
        if record_path.exists():
            saved=json.loads(record_path.read_text())
            if saved!={**record,'cache_sha256':digest(path)}:
                raise ValueError('Changed consumed attack evidence')
            with np.load(path,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['events'],item['events'])
                frames=data['attack_frames']
        else:
            samples,rate=sf.read(item['audio'],dtype='float32')
            if rate!=22050 or samples.ndim!=1 or len(samples)/rate!=item['duration']:
                raise ValueError('Changed regression audio clock')
            frames=sequences(observe(samples,rate),item['events'])
            np.savez_compressed(path,events=item['events'],attack_frames=frames)
            preserve(record_path,{**record,'cache_sha256':digest(path)})
        result.append({**item,'attack_frames':frames})
        records.append({**record,'cache_sha256':digest(path)})
        if len(result)%20==0:
            print(json.dumps({'regression_attacks_cached':len(result)}),flush=True)
    manifest={'plan_sha256':digest(folder/'plan.json'),'items':records,'consumed_regression':True}
    if (folder/'manifest.json').exists():
        if json.loads((folder/'manifest.json').read_text())!=manifest:
            raise ValueError('Changed completed consumed attack manifest')
    else:
        preserve(folder/'manifest.json',manifest)
    return result


def verify(run):
    report = run/'consumed-regression.json'
    if report.exists():
        raise ValueError('Preserve the frozen candidate regression result')
    plan,winner,network,normalizer = frozen(run)
    fine_plan=json.loads((Path(plan['baseline_root'])/'plan.json').read_text())
    parent=Path(fine_plan['parent_root'])
    items = load(parent,('test',))
    if len(items)!=296:
        raise ValueError('Incomplete consumed timing regression suite')
    guardian = BassHarmonic().models[1]
    items = attack_evidence(run,plan,winner,items)
    p = [probabilities(network,i['frames'],i['attack_frames'],i['x'],normalizer) for i in items]
    g = [guardian.probability(i['base_x']) for i in items]
    raw = evaluate(items,p,g,winner['timing_threshold'],winner['remove_threshold'])
    margin = evaluate(items,p,g,margin_threshold(winner['timing_threshold']),
                      margin_threshold(winner['remove_threshold']),.5)
    frozen(run)
    result = {'passes':bool(raw['passes'] and margin['passes']
                           and raw['onset_error_reduction_seconds']>1e-6
                           and margin['onset_error_reduction_seconds']>1e-6),
              'raw':raw,'margin':margin,'recordings':len(items),
              'checkpoint_sha256':winner['checkpoint_sha256'],'batch_selection_sha256':digest(run/'batch-selection.json'),
              'consumed_regression':True,'test_used_for_selection':False,'no_user_audio_or_scores':True,
              'scope':'All 296 previously consumed recordings. No claim of unseen human performance.'}
    preserve(report,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','margin')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
