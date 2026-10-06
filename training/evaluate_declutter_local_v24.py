"""Verify one frozen decluttering winner against all 360 consumed recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_harmonic import BassHarmonic
from bass_joint_timing_v18 import margin as margin_threshold
from bass_v16_data import load
from current_bass_v16_baseline import hashes
from declutter_local_model import probabilities
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import evaluate
from train_declutter_local_v24 import VERSION, checkpoint, contracts


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
            or any(not validation[k]['passes'] or validation[k]['false_notes_removed']<=0
                   for k in ('raw','margin'))
            or validation['raw']['guardian_threshold']!=winner['guardian_threshold']
            or validation['raw']['remove_threshold']!=winner['threshold']):
        raise ValueError('Changed selected timing checkpoint, evidence or thresholds')
    if (plan['required_consumed_regressions'] != 360 or len(plan['previous_fresh']) != 2
            or winner['threshold'] not in plan['thresholds']
            or winner['guardian_threshold'] not in plan['guardian_thresholds']):
        raise ValueError('Changed release partitions or selected thresholds')
    for name, threshold in (('raw', winner['threshold']), ('margin', margin_threshold(winner['threshold']))):
        row = validation[name]
        if (len(row['per_recording']) != 160 or len({i['id'] for i in row['per_recording']}) != 160
                or any(not i['passes'] for i in row['per_recording'])
                or (row['remove_threshold'], row['guardian_threshold']) != (threshold, winner['guardian_threshold'])
                or row['retimed_notes'] or abs(row['onset_error_reduction_seconds']) > 1e-9):
            raise ValueError('Changed complete validation gates or action configuration')
    network,normalizer = checkpoint(run/winner['checkpoint'])
    return plan,winner,network,normalizer


def attack_evidence(run, plan, winner, items):
    folder = Path(plan['baseline_root']).parent/'attack-local-bass-v19-v1'/'consumed-attack-features'
    manifest = json.loads((folder/'manifest.json').read_text())
    evidence_plan = json.loads((folder/'plan.json').read_text())
    if (digest(folder/'manifest.json') != plan['regression_attack_manifest_sha256']
            or digest(folder/'plan.json') != plan['regression_attack_plan_sha256']
            or manifest['plan_sha256'] != digest(folder/'plan.json')
            or not manifest['consumed_regression'] or evidence_plan['baseline_hashes'] != hashes()
            or not evidence_plan['no_fitting_selection_or_user_audio'] or evidence_plan['recordings'] != 296):
        raise ValueError('Changed sealed consumed attack observations')
    records = {r['id']:r for r in manifest['items']}
    if len(records) != 296:
        raise ValueError('Incomplete consumed observation identity')
    result = []
    for item in items:
        record = records.pop(item['id'])
        path = folder/(item['id']+'.npz')
        if (record['parent_cache_sha256'] != item['cache_sha256']
                or record['audio_sha256'] != item['audio_sha256'] or record['source_group'] != item['source_group']
                or record['cache_sha256'] != digest(path) or record['plan_sha256'] != digest(folder/'plan.json')):
            raise ValueError('Changed consumed attack source/cache')
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['events'], item['events'])
            frames = saved['attack_frames']
        if frames.shape != (len(item['events']),61,18) or not np.isfinite(frames).all() or np.any((frames<0)|(frames>1)):
            raise ValueError('Invalid sealed consumed attack features')
        result.append({**item,'attack_frames':frames})
    if records:
        raise ValueError('Missing consumed attack records')
    return result


def previous_fresh_observations(plan):
    folder=Path(plan['root'])
    manifest=json.loads((folder/'manifest.json').read_text())
    prepared=json.loads((folder/'plan.json').read_text())
    if (digest(folder/'manifest.json')!=plan['manifest_sha256']
            or digest(folder/'plan.json')!=plan['plan_sha256']
            or manifest['plan_sha256']!=digest(folder/'plan.json') or not manifest['now_consumed_regression']
            or not manifest['no_fitting_selection_user_audio_or_scores']
            or not prepared['no_fitting_selection_user_audio_or_scores']
            or prepared['baseline_hashes']!=hashes() or len(manifest['items'])!=32
            or len({r['id'] for r in manifest['items']})!=32):
        raise ValueError('Changed previously consumed procedural observation manifest')
    result=[]
    for record in manifest['items']:
        path=folder/record['id']/'features.npz'
        if digest(path)!=record['cache_sha256'] or record['group']!='test':
            raise ValueError('Changed previously consumed procedural features')
        with np.load(path,allow_pickle=False) as data:
            if str(data['plan_sha256'])!=manifest['plan_sha256']:
                raise ValueError('Changed previously consumed procedural identity')
            item={**record,**{k:data[k] for k in ('events','x','frames','attack_frames','base_x','eligible')}}
        n=len(item['events'])
        if (item['events'].shape!=(n,4) or item['x'].shape!=(n,84) or item['frames'].shape!=(n,40,9)
                or item['attack_frames'].shape!=(n,61,18) or item['base_x'].shape!=(n,52)
                or item['eligible'].shape!=(n,) or any(not np.isfinite(item[k]).all()
                    for k in ('events','x','frames','attack_frames','base_x'))
                or any(np.any((item[k]<0)|(item[k]>1)) for k in ('frames','attack_frames'))):
            raise ValueError('Invalid previously consumed procedural feature arrays')
        item.update(reference=np.asarray(record['reference'],float).reshape(-1,3),
                    pitch_reference=np.asarray(record['pitch_reference'],float).reshape(-1,3))
        result.append(item)
    return result


def verify(run):
    report = run/'consumed-regression.json'
    if report.exists():
        raise ValueError('Preserve the frozen candidate regression result')
    plan,winner,network,normalizer = frozen(run)
    fine_plan=json.loads((Path(plan['baseline_root'])/'plan.json').read_text())
    parent=Path(fine_plan['parent_root'])
    items = load(parent,('test',))
    for previous in plan['previous_fresh']:
        items.extend(previous_fresh_observations(previous))
    if len(items)!=360 or len({i['id'] for i in items})!=360:
        raise ValueError('Incomplete consumed timing regression suite')
    guardian = BassHarmonic().models[1]
    items = attack_evidence(run,plan,winner,items[:296])+items[296:]
    p = [probabilities(network,i['frames'],i['attack_frames'],i['x'],normalizer) for i in items]
    g = [guardian.probability(i['base_x']) for i in items]
    raw = evaluate(items,p,g,winner['threshold'],winner['guardian_threshold'])
    margin = evaluate(items,p,g,margin_threshold(winner['threshold']),winner['guardian_threshold'])
    frozen(run)
    result = {'passes':bool(raw['passes'] and margin['passes']
                           and raw['false_notes_removed']>0
                           and margin['false_notes_removed']>0),
              'raw':raw,'margin':margin,'recordings':len(items),
              'checkpoint_sha256':winner['checkpoint_sha256'],'batch_selection_sha256':digest(run/'batch-selection.json'),
              'consumed_regression':True,'test_used_for_selection':False,'no_user_audio_or_scores':True,
              'scope':'All 360 consumed recordings: original 296 plus V20 and V22 first-pass fixtures. No claim of unseen human performance.'}
    preserve(report,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','margin')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
