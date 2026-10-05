"""Fit bounded temporal CNN candidates; validation alone selects a frozen pair."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from app.services.temporal_note_model import (
    FEATURE_COUNT,
    OFFSETS,
    STEPS,
    VERSION,
    create_model,
    probability,
)
from bass_temporal_v15_data import collect
from current_bass_v15_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from train_bass_consensus import GUARDIANS, THRESHOLDS, matrix, select

STAGES = (10, 20, 30, 40, 60)
UPDATES = 40
BATCH_SIZE = 256
PROFILES = (
    {'name': 'temporal', 'width': 32, 'positive_weight': 12., 'weight_decay': .001},
    {'name': 'conservative', 'width': 48, 'positive_weight': 24., 'weight_decay': .002},
)
FRESH_SEEDS = tuple(range(263301, 263333))


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = ('training/train_bass_temporal_v16.py', 'training/bass_temporal_v15_data.py',
             'training/prepare_bass_v15_baseline.py', 'training/current_bass_v15_baseline.py',
             'training/prepare_vienna_bass_v15.py', 'training/slakh_references.py',
             'training/prepare_bass_positive_data.py', 'training/prepare_bass_texture_v12.py',
             'training/prepare_slakh_bass.py', 'training/prepare_slakh_training.py',
             'training/bass_temporal_v16_release.py',
             'training/bass_temporal_data.py', 'training/harmonic_candidate_features.py',
             'training/pitch_support_targets.py', 'training/pitch_interval_coverage.py',
             'training/cache_note_verifier.py', 'training/train_bass_consensus.py',
             'backend/app/services/temporal_note_model.py', 'backend/app/services/bass_temporal.py',
             'backend/app/services/bass_harmonic.py')
    return {name: digest(root / name) for name in names}


def checkpoint_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != VERSION or saved['width'] not in {p['width'] for p in PROFILES}:
        raise ValueError('Changed temporal architecture')
    model = create_model(saved['width'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, (saved['mean'].numpy(), saved['scale'].numpy())


def predictions(model, normalizer, items):
    return [probability(model, item['frames'], item['x'], normalizer) for item in items]


def supervised_arrays(items, positive_weight):
    # matrix() groups by corpus; temporal rows must use that identical order.
    ordered = sorted(items, key=lambda item: item['corpus'])
    frames = np.concatenate([item['frames'][item['eligible'] & item['mask']] for item in ordered])
    x, y, weights, counts = matrix(ordered, positive_weight)
    if len(x) != len(frames):
        raise ValueError('Temporal/context supervised ordering differs')
    return x, frames, y, weights, counts


def train(baseline, piano, output):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'code_sha256': contracts(),
            'baseline_root': str(baseline), 'baseline_plan_sha256': digest(baseline / 'plan.json'),
            'baseline_manifest_sha256': digest(baseline / 'manifest.json'),
            'piano_root': str(piano), 'piano_plan_sha256': digest(piano / 'plan.json'),
            'piano_manifest_sha256': digest(piano / 'manifest.json'), 'piano_audit_sha256': digest(piano / 'source-audit.json'),
            'profiles': list(PROFILES), 'stages': list(STAGES), 'updates_per_epoch': UPDATES,
            'batch_size': BATCH_SIZE, 'learning_rate': .0015, 'minimum_learning_rate': .00015,
            'threshold_grid': list(THRESHOLDS), 'guardian_grid': list(GUARDIANS), 'safety_factor': .5,
            'steps': STEPS, 'pitch_offsets': list(OFFSETS), 'context_features': FEATURE_COUNT,
            'guardian': 'Frozen V14 52-channel acoustic guardian; no new fitting, all probability features excluded from CNN inputs.',
            'fresh_seeds': list(FRESH_SEEDS), 'fresh_families': {'original': 16, 'texture': 16},
            'torch_version': str(torch.__version__), 'normalization': 'Training eligible/supervised context only; std floor .01; no frame normalizer or validation/test statistics.',
            'selection': 'Validation only: safe false-note gain, lower unweighted KEEP log loss, earlier epoch, fixed profile order.',
            'required_consumed_regressions': 252,
            'expanded_piano_regressions': 'All eligible reserved piano performers 19–22; acquire/evaluate only after frozen candidate and prior regression gates. All are consumed, not fresh human accuracy.',
            'first_pass_policy': 'Seeds 263301–263332 declared now; one winner only, no test-driven adjustment or replacement fresh check. Fresh fixture implementation must preserve existing original/texture generators and codec policy.',
            'gates': 'Every matched attack, offset-aware hold and supplied pitch interval preserved per recording; raw and half-margin thresholds; positive first-pass gain required.',
            'scope': 'CNN on actual V15 survivors with real-piano counterexamples, no user audio or scores; prior deletions cannot return. No production routing.'}
    preserve(output / 'plan.json', plan)
    training, validation = collect(baseline, piano)
    guardian = BassHarmonic().models[1]
    vg = [guardian.probability(item['base_x']) for item in validation]
    records = []
    for profile_index, profile in enumerate(PROFILES):
        destination = output / profile['name']
        destination.mkdir()
        seed = 263201 + profile_index
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        x, frames, y, weights, counts = supervised_arrays(training, profile['positive_weight'])
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
        tx, ty, tw = [torch.from_numpy(a.astype(np.float32)) for a in ((x - mean) / scale, y, weights)]
        tf = torch.from_numpy(frames)
        model = create_model(profile['width'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=profile['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(STAGES), eta_min=plan['minimum_learning_rate'])
        preserve(destination / 'run.json', {'seed': seed, 'profile': profile, 'training_counts': counts,
                 'training_events': len(y), 'training_recordings': len(training), 'validation_recordings': len(validation),
                 'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
        best, curves, memo = None, [], {}
        for epoch in range(1, max(STAGES) + 1):
            model.train()
            losses, started = [], time.monotonic()
            for _ in range(UPDATES):
                indices = torch.from_numpy(rng.integers(0, len(y), BATCH_SIZE))
                optimizer.zero_grad(set_to_none=True)
                logits = model(tf[indices], tx[indices])
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, ty[indices], reduction='none') * tw[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite temporal training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = {'epoch': epoch, 'training_loss': float(np.mean(losses)), 'seconds': time.monotonic() - started}
            if epoch in STAGES:
                probabilities = predictions(model, (mean, scale), validation)
                vy = np.concatenate([item['y'][item['eligible'] & item['mask']] for item in validation])
                vp = np.concatenate([p[item['eligible'] & item['mask']] for item, p in zip(validation, probabilities, strict=True)])
                clipped = np.clip(vp.astype(float), 1e-7, 1 - 1e-7)
                val_loss = float(-np.mean(vy * np.log(clipped) + (1 - vy) * np.log(1 - clipped)))
                chosen, search = select(validation, probabilities, vg, memo)
                row.update(validation_log_loss=val_loss, selected=chosen is not None,
                           false_notes_removed=chosen['false_notes_removed'] if chosen else 0)
                preserve(destination / f'search-{epoch:03d}.json', search)
                path = destination / f'epoch-{epoch:03d}.pt'
                torch.save({'version': VERSION, 'width': profile['width'], 'epoch': epoch,
                            'state_dict': model.state_dict(), 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale)}, path)
                if chosen:
                    key = (chosen['false_notes_removed'], -val_loss, -epoch)
                    if best is None or key > best[0]:
                        best = (key, path, chosen, epoch, val_loss)
            curves.append(row)
            if epoch % 5 == 0:
                print(json.dumps({'profile': profile['name'], **row}), flush=True)
        preserve(destination / 'learning-curves.json', curves)
        if best:
            preserve(destination / 'validation.json', best[2])
        selection = {'name': profile['name'], 'selected': best is not None,
                     'epoch': best[3] if best else None, 'checkpoint': str(best[1].relative_to(output)) if best else None,
                     'checkpoint_sha256': digest(best[1]) if best else None,
                     'validation_sha256': digest(destination / 'validation.json') if best else None,
                     'validation_false_notes_removed': best[2]['false_notes_removed'] if best else 0,
                     'threshold': best[2]['threshold'] if best else None,
                     'guardian_threshold': best[2]['guardian_threshold'] if best else None,
                     'validation_log_loss': best[4] if best else None,
                     'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False}
        preserve(destination / 'selection.json', selection)
        records.append(selection)
    eligible = [row for row in records if row['selected']]
    winner = max(eligible, key=lambda row: (row['validation_false_notes_removed'], -row['validation_log_loss'], -row['epoch'])) if eligible else None
    preserve(output / 'batch-selection.json', {'selected': winner is not None, 'winner': winner, 'candidates': records,
             'plan_sha256': digest(output / 'plan.json'), 'test_used_for_selection': False})
    print(json.dumps({'winner': winner}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('piano', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    train(args.baseline, args.piano, args.output)
