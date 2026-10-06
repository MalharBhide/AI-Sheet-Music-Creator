"""Fit binary acoustic support on V16 survivors; select only on validation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from attack_local_bass_data import load
from bass_temporal_data import annotate
from current_bass_v16_baseline import hashes
from declutter_local_model import VERSION, model, probabilities, warm_start
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import choose
from train_early_consensus_v22 import contracts as prior_contracts

STAGES = (20, 40, 60, 80)
UPDATES, BATCH_SIZE = 40, 256
THRESHOLDS = (.6, .7, .8, .9, .95, .99)
GUARDIANS = (.15, .3, .5)
PROFILES = (
    {'name': 'support-conservative', 'width': 24, 'positive_weight': 12., 'seed': 268001},
    {'name': 'support-protected', 'width': 32, 'positive_weight': 24., 'seed': 268002},
)
FRESH_SEEDS = tuple(range(268101, 268133))
WARM_SHA = '993095e4ecaae7d9f9cda01042860442f81aaddc9b5c9be94ffae9b5307c302e'


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = ('training/train_declutter_local_v23.py', 'training/evaluate_declutter_local_v23.py',
             'training/fresh_declutter_local_v23.py', 'training/score_declutter_local.py',
             'training/declutter_local_model.py')
    return {**prior_contracts(), **{name: digest(root / name) for name in names}}


def supervised(items, positive_weight):
    xs, frames, fine, labels, weights, counts = [], [], [], [], [], {}
    corpora = sorted({i['corpus'] for i in items})
    for corpus in corpora:
        group = [i for i in items if i['corpus'] == corpus and np.any(i['mask'] & i['eligible'])]
        counts[corpus] = {'recordings': len(group), 'class_counts': [0, 0]}
        for item in group:
            selected = item['mask'] & item['eligible']
            y = (1 - item['y'][selected]).astype(np.int64)
            xs.append(item['x'][selected])
            frames.append(item['frames'][selected])
            fine.append(item['attack_frames'][selected])
            labels.append(y)
            weights.append(np.where(y == 0, positive_weight, 1.) / (len(corpora)*len(group)*len(y)))
            counts[corpus]['class_counts'] = (np.asarray(counts[corpus]['class_counts'])
                                            + np.bincount(y, minlength=2)).tolist()
    x, f, af, y, w = map(np.concatenate, (xs, frames, fine, labels, weights))
    return x, f, af, y, w/w.mean(), counts


def checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != VERSION or saved['width'] not in (24, 32):
        raise ValueError('Changed binary support architecture')
    network = model(saved['width'])
    network.load_state_dict(saved['state_dict'], strict=True)
    return network, (saved['mean'].numpy(), saved['scale'].numpy())


def selection_key(gain, margin_gain, loss, epoch):
    if not np.isfinite([gain, margin_gain, loss, epoch]).all():
        raise ValueError('Invalid validation selection metrics')
    return gain, margin_gain, -loss, -epoch


def train(baseline, warm, output):
    if digest(warm) != WARM_SHA:
        raise ValueError('Changed deployed V16 warm-start encoder')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    regression = baseline.parent / 'attack-local-bass-v19-v1' / 'consumed-attack-features'
    previous = [baseline.parent / name for name in ('pitch-consensus-v20-fresh-v1', 'early-consensus-v22-fresh-v1')]
    plan = {'version': VERSION, 'code_sha256': contracts(), 'baseline_hashes': hashes(),
            'baseline_root': str(baseline), 'baseline_manifest_sha256': digest(baseline/'manifest.json'),
            'baseline_plan_sha256': digest(baseline/'plan.json'),
            'warm_start_sha256': WARM_SHA, 'warm_start': str(warm),
            'regression_attack_manifest_sha256': digest(regression/'manifest.json'),
            'regression_attack_plan_sha256': digest(regression/'plan.json'),
            'previous_fresh': [{'root': str(p), 'manifest_sha256': digest(p/'manifest.json'),
                                'plan_sha256': digest(p/'plan.json')} for p in previous],
            'profiles': list(PROFILES), 'stages': list(STAGES), 'updates_per_epoch': UPDATES,
            'batch_size': BATCH_SIZE, 'learning_rate': .0006, 'minimum_learning_rate': .00006,
            'weight_decay': .002, 'thresholds': list(THRESHOLDS), 'guardian_thresholds': list(GUARDIANS),
            'architecture': 'Coarse 40x9 CQT + local 61x18 CQT/STFT + 84 context, binary KEEP/REMOVE head.',
            'labels': 'Existing pitch-coverage, attack and hold-protected supervision only. Ambiguous candidates excluded. No timing labels or synthetic onset changes.',
            'normalizer': 'Eligible supervised training context only; standard deviation floor .01.',
            'sampling': 'Equal corpus/recording loss before KEEP-class protection; uniform event batches.',
            'selection': '160 validation recordings only: positive false-note reduction under raw and stricter thresholds, all preservation gates; then raw gain, margin gain, lower unweighted log loss, earlier epoch, fixed profile order.',
            'gates': 'Per recording retain every previously matched attack and hold, all reference pitch interval coverage; no added false positives. Retained events are byte-identical, including all clocks/pitches/velocities. Fixed-pair timing/repeat-spacing error cannot increase; deletion earns no timing credit.',
            'margin': 'Removal confidence t+(1-t)/2; acoustic guardian unchanged.',
            'required_consumed_regressions': 360, 'fresh_seeds': list(FRESH_SEEDS),
            'fresh_families': {'original': 16, 'texture': 16},
            'first_pass_policy': 'One frozen validation winner; positive false-note reduction for raw and margin plus every preservation gate. No test tuning, replacement checkpoint or replacement seeds.',
            'rights': 'Existing Vienna CC BY 4.0, GuitarSet CC BY 4.0, BabySlakh CC BY 4.0 and project-authored signals; existing V16 notices apply.',
            'test_used_for_selection': False, 'no_user_audio_or_scores': True,
            'torch_version': str(torch.__version__),
            'scope': 'Acoustic unsupported-note classification on actual V16 survivors. No new notes, timing correction, website routing or accuracy claim on arbitrary songs.'}
    preserve(output/'plan.json', plan)
    raw = load(baseline)
    training = [annotate(i) for i in raw if i['group'] == 'train']
    validation = [annotate(i) for i in raw if i['group'] == 'validation']
    if len(training) != 573 or len(validation) != 160 or any(i['group'] == 'test' for i in raw):
        raise ValueError('Invalid fitting/selection partitions')
    guardian = BassHarmonic().models[1]
    guards = [guardian.probability(i['base_x']) for i in validation]
    selections = []
    for profile in PROFILES:
        folder = output/profile['name']
        folder.mkdir()
        torch.manual_seed(profile['seed'])
        rng = np.random.default_rng(profile['seed'])
        x, f, af, y, w, counts = supervised(training, profile['positive_weight'])
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
        tx, tf, ta, ty, tw = (torch.from_numpy(a) for a in (
            ((x-mean)/scale).astype(np.float32), f.astype(np.float32), af.astype(np.float32),
            y.astype(np.int64), w.astype(np.float32)))
        saved = torch.load(warm, map_location='cpu', weights_only=True)
        network = warm_start(saved, profile['width'])
        optimizer = torch.optim.AdamW(network.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(STAGES), eta_min=plan['minimum_learning_rate'])
        preserve(folder/'run.json', {'profile': profile, 'plan_sha256': digest(output/'plan.json'),
                 'training_counts': counts, 'training_events': len(y), 'training_recordings': 573,
                 'validation_recordings': 160, 'test_used_for_selection': False})
        best, curves, memo = None, [], {}
        for epoch in range(1, max(STAGES)+1):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(UPDATES):
                indices = torch.from_numpy(rng.integers(0, len(y), BATCH_SIZE))
                optimizer.zero_grad(set_to_none=True)
                logits = network(tf[indices], ta[indices], tx[indices])
                loss = (torch.nn.functional.cross_entropy(logits, ty[indices], reduction='none') * tw[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite binary support loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.monotonic()-started}
            if epoch in STAGES:
                ps = [probabilities(network, i['frames'], i['attack_frames'], i['x'], (mean, scale)) for i in validation]
                vy = np.concatenate([(1-i['y'][i['eligible'] & i['mask']]).astype(int) for i in validation])
                vp = np.concatenate([p[i['eligible'] & i['mask']] for i, p in zip(validation, ps, strict=True)])
                val_loss = float(-np.log(np.clip(vp[np.arange(len(vy)), vy], 1e-7, 1.)).mean())
                chosen, search = choose(validation, ps, guards, THRESHOLDS, GUARDIANS, memo)
                preserve(folder/f'search-{epoch:03d}.json', search)
                path = folder/f'epoch-{epoch:03d}.pt'
                torch.save({'version': VERSION, 'width': profile['width'], 'epoch': epoch,
                            'state_dict': network.state_dict(), 'mean': torch.from_numpy(mean),
                            'scale': torch.from_numpy(scale)}, path)
                row.update(validation_log_loss=val_loss, selected=chosen is not None,
                           false_notes_removed=chosen['raw']['false_notes_removed'] if chosen else 0)
                if chosen:
                    key = selection_key(chosen['raw']['false_notes_removed'], chosen['margin']['false_notes_removed'], val_loss, epoch)
                    if best is None or key > best[0]:
                        best = (key, path, chosen, epoch, val_loss)
            curves.append(row)
            if epoch % 5 == 0:
                print(json.dumps({'profile': profile['name'], **row}), flush=True)
        preserve(folder/'learning-curves.json', curves)
        if best:
            preserve(folder/'validation.json', best[2])
        selected = {'profile': profile['name'], 'width': profile['width'], 'selected': best is not None,
                    'epoch': best[3] if best else None, 'checkpoint': str(best[1].relative_to(output)) if best else None,
                    'checkpoint_sha256': digest(best[1]) if best else None,
                    'validation_sha256': digest(folder/'validation.json') if best else None,
                    'false_notes_removed': best[0][0] if best else 0,
                    'margin_false_notes_removed': best[0][1] if best else 0,
                    'validation_log_loss': best[4] if best else None,
                    'threshold': best[2]['raw']['remove_threshold'] if best else None,
                    'guardian_threshold': best[2]['raw']['guardian_threshold'] if best else None,
                    'plan_sha256': digest(output/'plan.json'), 'test_used_for_selection': False}
        preserve(folder/'selection.json', selected)
        selections.append(selected)
    eligible = [s for s in selections if s['selected']]
    winner = max(eligible, key=lambda s: selection_key(s['false_notes_removed'], s['margin_false_notes_removed'], s['validation_log_loss'], s['epoch'])) if eligible else None
    if plan['code_sha256'] != contracts() or plan['baseline_hashes'] != hashes():
        raise ValueError('Source/production baseline changed during fitting')
    preserve(output/'batch-selection.json', {'selected': winner is not None, 'winner': winner,
             'candidates': selections, 'plan_sha256': digest(output/'plan.json'),
             'test_used_for_selection': False, 'no_user_audio_or_scores': True})
    print(json.dumps({'winner': winner}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('warm', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    train(args.baseline, args.warm, args.output)
