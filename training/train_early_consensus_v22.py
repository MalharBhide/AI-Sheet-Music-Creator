"""Train guarded timing/decluttering CNNs on the current V16 survivors."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from attack_local_bass_data import load
from attack_local_model import VERSION as MODEL_VERSION
from attack_local_model import model, probabilities
from attack_local_model import warm_start as initialize
from current_bass_v16_baseline import hashes
from pitch_consensus_early import targets
from prepare_robust_training_stems import digest, preserve
from score_early_consensus import choose

VERSION = 'bass-v16-early-consensus-attack-v22-v1'
STAGES = (20,40,60,80)
UPDATES, BATCH_SIZE = 40, 256
THRESHOLDS = (.8,.9,.95,.99,1.01)
PROFILES = (
    {'name':'early-conservative','width':24,'class_weights':[24.,6.,1.,1.], 'seed':267001},
    {'name':'early-supported','width':32,'class_weights':[36.,6.,1.,1.], 'seed':267002},
)
FRESH_SEEDS = tuple(range(267101,267133))
WARM_SHA = '993095e4ecaae7d9f9cda01042860442f81aaddc9b5c9be94ffae9b5307c302e'


def selection_key(onset_gain, false_notes_removed, validation_log_loss, epoch):
    if not np.isfinite([onset_gain, false_notes_removed, validation_log_loss, epoch]).all():
        raise ValueError('Nonfinite validation selection metric')
    return round(onset_gain,9), false_notes_removed, -validation_log_loss, -epoch


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = ('training/pitch_consensus_early.py','training/score_early_consensus.py',
             'training/train_early_consensus_v22.py','training/evaluate_early_consensus_v22.py',
             'training/fresh_early_consensus_v22.py',
             'training/train_pitch_consensus_v21.py','training/evaluate_pitch_consensus_v21.py',
             'training/fresh_pitch_consensus_v21.py',
             'training/train_pitch_consensus_v20.py','training/evaluate_pitch_consensus_v20.py',
             'training/pitch_consensus_timing.py','training/score_pitch_consensus_timing.py',
             'training/fresh_pitch_consensus_v20.py',
             'training/train_attack_local_bass_v19.py','training/evaluate_attack_local_bass_v19.py',
             'training/prepare_bass_positive_data.py','training/prepare_bass_texture_v12.py','training/prepare_slakh_bass.py',
             'training/attack_local_bass_data.py','training/attack_local_features.py','training/attack_local_model.py',
             'training/prepare_attack_local_bass.py','training/train_bass_joint_timing_v18.py',
             'training/bass_joint_timing_v18.py','training/bass_joint_timing.py',
             'training/score_bass_joint_timing.py','training/bass_v16_data.py',
             'training/current_bass_v16_baseline.py','training/prepare_bass_v16_baseline.py',
             'training/bass_temporal_data.py','training/bass_residual_coverage_data.py',
             'training/pitch_support_targets.py','training/pitch_interval_coverage.py',
             'training/train_melody_timing.py','training/train_note_verifier.py',
             'backend/app/services/temporal_note_model.py','backend/app/services/bass_harmonic.py')
    return {name:digest(root/name) for name in names}


def supervised(items, class_weights):
    xs, frames, fine, labels, weights, counts = [], [], [], [], [], {}
    corpora = sorted({i['corpus'] for i in items})
    for corpus in corpora:
        group = [i for i in items if i['corpus'] == corpus and np.any(i['mask'] & i['eligible'])]
        counts[corpus] = {'recordings':len(group),'class_counts':[0,0,0,0]}
        for item in group:
            selected = item['mask'] & item['eligible']
            y = item['action_y'][selected]
            xs.append(item['x'][selected])
            frames.append(item['frames'][selected])
            fine.append(item['attack_frames'][selected])
            labels.append(y)
            weights.append(np.asarray(class_weights)[y]/(len(corpora)*len(group)*len(y)))
            counts[corpus]['class_counts'] = (np.asarray(counts[corpus]['class_counts'])
                                            +np.bincount(y,minlength=4)).tolist()
    x, f, y, w = map(np.concatenate,(xs,frames,labels,weights))
    return x, f, np.concatenate(fine), y, w/w.mean(), counts


def warm_start(path, width):
    if digest(path) != WARM_SHA:
        raise ValueError('Changed deployed V16 encoder checkpoint')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    return initialize(model(width),saved)


def checkpoint(path):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['version'] != VERSION or saved['model_version'] != MODEL_VERSION or saved['width'] not in (24,32):
        raise ValueError('Changed joint timing architecture')
    network = model(saved['width'])
    network.load_state_dict(saved['state_dict'],strict=True)
    return network,(saved['mean'].numpy(),saved['scale'].numpy())


def train(baseline, warm, output):
    if digest(warm) != WARM_SHA:
        raise ValueError('Changed warm-start checkpoint')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    regression = baseline.parent/'attack-local-bass-v19-v1'/'consumed-attack-features'
    plan = {'version':VERSION,'model_version':MODEL_VERSION,
            'regression_attack_manifest_sha256':digest(regression/'manifest.json'),
            'previous_fresh_root':str(baseline.parent/'pitch-consensus-v20-fresh-v1'),
            'previous_fresh_manifest_sha256':digest(baseline.parent/'pitch-consensus-v20-fresh-v1'/'manifest.json'),
            'previous_fresh_plan_sha256':digest(baseline.parent/'pitch-consensus-v20-fresh-v1'/'plan.json'),
            'selection_gain_rounding_decimals':9,
            'imbalance_policy':'Two predeclared moderate early-label loss weights; late labels are unavailable and their formerly positive examples become KEEP. Counts and normalization use training only.',
            'regression_attack_plan_sha256':digest(regression/'plan.json'),
            'regression_observation_policy':'Previously sealed observed features only; record hashes are pinned before fitting; no regression labels, waveform, metrics or predictions enter fitting/selection.','baseline_hashes':hashes(),'code_sha256':contracts(),
            'baseline_root':str(baseline),'baseline_plan_sha256':digest(baseline/'plan.json'),
            'baseline_manifest_sha256':digest(baseline/'manifest.json'),
            'warm_start':str(warm),'warm_start_sha256':WARM_SHA,'profiles':list(PROFILES),
            'coarse_width':48,'attack_shape':[61,18],'stages':list(STAGES),'updates_per_epoch':UPDATES,'batch_size':BATCH_SIZE,
            'learning_rate':.0006,'minimum_learning_rate':.00006,'weight_decay':.002,
            'thresholds':list(THRESHOLDS),'margin_confidence':'t+(1-t)/2; disabled 1.01 remains disabled','guardian_threshold':.3,'shifts_seconds':[-.02,0.,0.,0.],
            'normalizer':'Mean/std of supervised eligible training context only; std floor .01.',
            'sampling':'Equal corpus and recording loss before class weights; uniform training event batches.',
            'selection':'Onset gains rounded to 9 decimal seconds to prevent floating-point noise breaking metric ties; 160 validation recordings only: positive raw and half-strength margin onset error reduction, then false-note reduction, then lower class log loss, earlier epoch, fixed profile order.',
            'gates':'Per recording: no lost matched attacks/holds or reference pitch coverage; no extra false positives; onset MAE, release MAE and repeated-key interval MAE must not increase. Fixed 200ms baseline same-pitch pairs; deletion earns no timing credit.',
            'test_used_for_selection':False,'required_consumed_regressions':328,
            'fresh_seeds':list(FRESH_SEEDS),'fresh_families':{'original':16,'texture':16},
            'first_pass_policy':'One validation-selected frozen winner only; no test-driven tuning or replacement seeds; first pass must improve timing and preserve quality.',
            'torch_version':str(torch.__version__),'no_user_audio_or_scores':True,
            'scope':'Early-only dual-scale CNN trained with same-key consensus labels on actual V16 survivors. Every occurrence of a piano key must confidently agree before moving its attacks uniformly earlier; releases, pitches, velocities and all repeated-note intervals remain unchanged. A deletion on that key disables timing correction. No website routing.',
            'label_policy':'One uniform early pitch-group correction only if onset error decreases by more than .018 seconds per event and all unchanged per-recording gates pass. All group occurrences must be eligible protected positive training notes. No regression data in labels.',
            'decoder_policy':'LATE is always KEEP; retained intervals never shrink. Retain every release exactly; same-key occurrences shift by the same bounded 20ms or 10ms amount, with no new overlaps and at least 45ms durations/attack spacing. This addresses systematic onset bias and preserves cadence, not incorrect spacing itself.'}
    preserve(output/'plan.json',plan)
    raw = load(baseline)
    training = [targets(i) for i in raw if i['group']=='train']
    validation = [targets(i) for i in raw if i['group']=='validation']
    if len(training)!=573 or len(validation)!=160 or any(i['group']=='test' for i in raw):
        raise ValueError('Invalid fitting/selection partitions')
    guardian = BassHarmonic().models[1]
    guards = [guardian.probability(i['base_x']) for i in validation]
    selections = []
    for profile in PROFILES:
        folder = output/profile['name']
        folder.mkdir()
        torch.manual_seed(profile['seed'])
        rng = np.random.default_rng(profile['seed'])
        x,f,af,y,w,counts = supervised(training,profile['class_weights'])
        mean,scale = x.mean(axis=0),np.maximum(x.std(axis=0),.01)
        tx,tf,ta,ty,tw = (torch.from_numpy(a) for a in (((x-mean)/scale).astype(np.float32),
                       f.astype(np.float32),af.astype(np.float32),y.astype(np.int64),w.astype(np.float32)))
        network = warm_start(warm,profile['width'])
        optimizer = torch.optim.AdamW(network.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(STAGES),eta_min=plan['minimum_learning_rate'])
        run = {'profile':profile,'plan_sha256':digest(output/'plan.json'),'training_counts':counts,
               'training_events':len(y),'training_recordings':len(training),'validation_recordings':len(validation),
               'test_used_for_selection':False}
        preserve(folder/'run.json',run)
        print(json.dumps(run),flush=True)
        best,curves,memo = None,[],{}
        for epoch in range(1,max(STAGES)+1):
            network.train()
            losses,started = [],time.monotonic()
            for _ in range(UPDATES):
                indices = torch.from_numpy(rng.integers(0,len(y),BATCH_SIZE))
                optimizer.zero_grad(set_to_none=True)
                logits = network(tf[indices],ta[indices],tx[indices])
                loss = (torch.nn.functional.cross_entropy(logits,ty[indices],reduction='none')*tw[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite joint timing loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = {'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.monotonic()-started}
            if epoch in STAGES:
                ps = [probabilities(network,i['frames'],i['attack_frames'],i['x'],(mean,scale)) for i in validation]
                vy = np.concatenate([i['action_y'][i['eligible'] & i['mask']] for i in validation])
                vp = np.concatenate([p[i['eligible'] & i['mask']] for i,p in zip(validation,ps,strict=True)])
                val_loss = float(-np.log(np.clip(vp[np.arange(len(vy)),vy],1e-7,1.)).mean())
                chosen,search = choose(validation,ps,guards,THRESHOLDS,memo)
                preserve(folder/f'search-{epoch:03d}.json',search)
                path = folder/f'epoch-{epoch:03d}.pt'
                torch.save({'version':VERSION,'model_version':MODEL_VERSION,'width':profile['width'],'epoch':epoch,'state_dict':network.state_dict(),
                            'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale)},path)
                row.update(validation_log_loss=val_loss,selected=chosen is not None,
                           onset_gain=chosen['raw']['onset_error_reduction_seconds'] if chosen else 0.)
                if chosen:
                    key = selection_key(chosen['raw']['onset_error_reduction_seconds'],chosen['raw']['false_notes_removed'],val_loss,epoch)
                    if best is None or key>best[0]:
                        best = (key,path,chosen,epoch,val_loss)
            curves.append(row)
            if epoch%5==0:
                print(json.dumps({'profile':profile['name'],**row}),flush=True)
        preserve(folder/'learning-curves.json',curves)
        if best:
            preserve(folder/'validation.json',best[2])
        selected = {'profile':profile['name'],'width':profile['width'],'selected':best is not None,'epoch':best[3] if best else None,
                    'checkpoint':str(best[1].relative_to(output)) if best else None,
                    'checkpoint_sha256':digest(best[1]) if best else None,
                    'validation_sha256':digest(folder/'validation.json') if best else None,
                    'onset_gain':best[2]['raw']['onset_error_reduction_seconds'] if best else 0.,'false_notes_removed':best[0][1] if best else 0,
                    'validation_log_loss':best[4] if best else None,
                    'timing_threshold':best[2]['raw']['timing_threshold'] if best else None,
                    'remove_threshold':best[2]['raw']['remove_threshold'] if best else None,
                    'plan_sha256':digest(output/'plan.json'),'test_used_for_selection':False}
        preserve(folder/'selection.json',selected)
        selections.append(selected)
    eligible = [s for s in selections if s['selected']]
    winner = max(eligible,key=lambda s:selection_key(s['onset_gain'],s['false_notes_removed'],s['validation_log_loss'],s['epoch'])) if eligible else None
    if plan['code_sha256']!=contracts() or plan['baseline_hashes']!=hashes():
        raise ValueError('Source/production baseline changed during fitting')
    preserve(output/'batch-selection.json',{'selected':winner is not None,'winner':winner,'candidates':selections,
             'plan_sha256':digest(output/'plan.json'),'test_used_for_selection':False,'no_user_audio_or_scores':True})
    print(json.dumps({'winner':winner},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('warm',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    train(args.baseline,args.warm,args.output)
