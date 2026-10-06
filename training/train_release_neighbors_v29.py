"""Train warm attack/ending encoders with a neutral recurrence branch on V25 survivors."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from current_bass_v25_baseline import hashes
from fragment_support_model import targets
from groove_bass_data import load
from prepare_robust_training_stems import digest, preserve
from release_neighbor_model import VERSION, WARM_SHA, initialize, model, probabilities
from score_declutter_local import choose
from train_attack_release_v25 import supervised as acoustic_supervised

STAGES = (20, 40, 60, 80)
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
GUARDIANS = (0.15, 0.3, 0.5)
PROFILES = (
    {
        "name": "neighbor-conservative",
        "width": 32,
        "positive_weight": 16.0,
        "fragment_weight": 8.0,
        "seed": 274001,
    },
    {
        "name": "neighbor-protected",
        "width": 64,
        "positive_weight": 32.0,
        "fragment_weight": 16.0,
        "seed": 274002,
    },
)


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/train_release_neighbors_v29.py",
        "training/groove_bass_data.py",
        "training/fusion_bass_data.py",
        "training/fragment_support_model.py",
        "training/prepare_groove_bass.py",
        "training/acquire_groove_subset.py",
        "training/release_neighbor_features.py",
        "training/release_neighbor_model.py",
        "training/fusion_model.py",
        "training/pitch_interval_coverage.py",
        "training/periodicity_features.py",
        "training/score_declutter_local.py",
        "training/score_early_consensus.py",
        "training/bass_joint_timing.py",
        "training/partial_hold_training.py",
        "training/evaluate_release_neighbors_v29.py",
        "training/fresh_release_neighbors_v29.py",
        "training/train_attack_release_v25.py",
        "training/train_fusion_v27.py",
    )
    return {name: digest(root / name) for name in names}


def supervised(items, profile):
    x, frames, attacks, endings, labels, weights, counts = acoustic_supervised(
        items, profile["positive_weight"], profile["fragment_weight"]
    )
    phases, subtype_labels = [], []
    for corpus in sorted({i["corpus"] for i in items}):
        group = [i for i in items if i["corpus"] == corpus and np.any(i["mask"] & i["eligible"])]
        phases.extend(
            np.column_stack((item["periodicity"], item["release_neighbors"]))[
                item["mask"] & item["eligible"]
            ]
            for item in group
        )
        subtype_labels.extend(targets(item)[item["mask"] & item["eligible"]] for item in group)
    phase = np.concatenate(phases)
    if len(phase) != len(x):
        raise ValueError("Acoustic/recurrence supervision misaligned")
    y = np.concatenate(subtype_labels)
    if len(y) != len(labels) or not np.array_equal(y == 2, labels == 1):
        raise ValueError("Changed support subtype event alignment")
    counts["support_subtypes"] = np.bincount(y, minlength=3).tolist()
    return x, frames, attacks, endings, phase, y, weights, counts


def train(folder, mixtures, warm, output):
    if digest(warm) != WARM_SHA:
        raise ValueError("Changed V25 warm-start checkpoint")
    raw = load(folder, mixtures)
    training = [i for i in raw if i["group"] == "train"]
    validation = [i for i in raw if i["group"] == "validation"]
    if len(training) != 702 or len(validation) != 206:
        raise ValueError("Incorrect V29 fitting partition")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "baseline_root": str(folder),
        "mixtures_root": str(mixtures),
        "mixtures_manifest_sha256": digest(mixtures / "manifest.json"),
        "mixtures_plan_sha256": digest(mixtures / "plan.json"),
        "baseline_manifest_sha256": digest(folder / "manifest.json"),
        "warm_start": str(warm),
        "warm_start_sha256": WARM_SHA,
        "profiles": list(PROFILES),
        "stages": list(STAGES),
        "updates_per_epoch": 40,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "minimum_learning_rate": 0.00003,
        "weight_decay": 0.002,
        "thresholds": list(THRESHOLDS),
        "guardian_thresholds": list(GUARDIANS),
        "architecture": "Warm V25 coarse/attack/ending encoders + 84 context, learned 63-feature waveform-recurrence plus eight observed same-pitch release-neighbor features; partial/other protected/unsupported support taxonomy, protected masses summed at inference",
        "support_taxonomy": {
            "classes": ["partial_support", "other_protected_support", "unsupported"],
            "partial_fraction_below": 0.25,
            "positive_support_seconds_above": 1e-6,
            "training_only": True,
            "no_target_as_input": True,
            "initial_protected_mass_equals_v25": True,
        },
        "required_consumed_regressions": 392,
        "periodicity_observation_root": str(
            folder.parent / "periodicity-v26-v1/consumed-periodicity"
        ),
        "periodicity_observation_manifest_sha256": digest(
            folder.parent / "periodicity-v26-v1/consumed-periodicity/manifest.json"
        ),
        "periodicity_observation_plan_sha256": digest(
            folder.parent / "periodicity-v26-v1/consumed-periodicity/plan.json"
        ),
        "fresh_policy": "Only after both frozen confidence settings pass every consumed regression. All four reserved NSynth instruments, six declared MP3 sequences each; three source clips from two reserved GMD performers, seven sequences each using only reserved NSynth instruments; plus 32 new project-authored clips with seeds 274101–274132. Total 77 first-pass recordings. Both settings must reduce false notes and preserve all attacks, holds, pitch coverage and fixed-pair timing.",
        "new_first_pass_seeds": list(range(274101, 274133)),
        "release_requires_portable_parity_and_native_checks": True,
        "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True,
    }
    preserve(output / "plan.json", plan)
    selections = []
    guardian = BassHarmonic().models[1]
    guards = [guardian.probability(i["base_x"]) for i in validation]
    for profile in PROFILES:
        directory = output / profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        x, f, a, e, phase, y, w, counts = supervised(training, profile)
        saved = torch.load(warm, map_location="cpu", weights_only=True)
        mean = np.concatenate((saved["mean"].numpy(), phase.mean(axis=0)))
        scale = np.concatenate((saved["scale"].numpy(), np.maximum(phase.std(axis=0), 0.01)))
        tx, tf, ta, te, tp, ty, tw = [
            torch.from_numpy(v)
            for v in (
                ((x - mean[:84]) / scale[:84]).astype(np.float32),
                f.astype(np.float32),
                a.astype(np.float32),
                e.astype(np.float32),
                ((phase - mean[84:]) / scale[84:]).astype(np.float32),
                y,
                w.astype(np.float32),
            )
        ]
        network = initialize(model(profile["width"]), saved)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, weight_decay=0.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.00003)
        preserve(
            directory / "run.json",
            {
                "profile": profile,
                "plan_sha256": digest(output / "plan.json"),
                "training_counts": counts,
                "training_events": len(y),
                "training_recordings": 702,
                "validation_recordings": 206,
                "test_used_for_selection": False,
            },
        )
        best = None
        curves = []
        memo = {}
        for epoch in range(1, 81):
            network.train()
            losses = []
            started = time.monotonic()
            for _ in range(40):
                indices = torch.from_numpy(rng.integers(0, len(y), 256))
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    torch.nn.functional.cross_entropy(
                        network(tf[indices], ta[indices], te[indices], tx[indices], tp[indices]),
                        ty[indices],
                        reduction="none",
                    )
                    * tw[indices]
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite recurrence loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "seconds": time.monotonic() - started,
            }
            if epoch in STAGES:
                ps = [probabilities(network, i, (mean, scale)) for i in validation]
                chosen, search = choose(validation, ps, guards, THRESHOLDS, GUARDIANS, memo)
                preserve(directory / f"search-{epoch:03d}.json", search)
                checkpoint = directory / f"epoch-{epoch:03d}.pt"
                torch.save(
                    {
                        "version": VERSION,
                        "width": profile["width"],
                        "epoch": epoch,
                        "state_dict": network.state_dict(),
                        "mean": torch.from_numpy(mean),
                        "scale": torch.from_numpy(scale),
                    },
                    checkpoint,
                )
                row.update(
                    selected=chosen is not None,
                    false_notes_removed=chosen["raw"]["false_notes_removed"] if chosen else 0,
                )
                if chosen:
                    key = (
                        chosen["raw"]["false_notes_removed"],
                        chosen["margin"]["false_notes_removed"],
                        -epoch,
                    )
                    if best is None or key > best[0]:
                        best = (
                            key,
                            {
                                "profile": profile["name"],
                                "width": profile["width"],
                                "epoch": epoch,
                                "checkpoint": str(checkpoint.relative_to(output)),
                                "checkpoint_sha256": digest(checkpoint),
                                "threshold": chosen["raw"]["remove_threshold"],
                                "guardian_threshold": chosen["raw"]["guardian_threshold"],
                                "false_notes_removed": key[0],
                                "margin_false_notes_removed": key[1],
                                "plan_sha256": digest(output / "plan.json"),
                                "test_used_for_selection": False,
                            },
                            chosen,
                        )
            curves.append(row)
            print(json.dumps({"profile": profile["name"], **row}), flush=True)
        preserve(directory / "learning-curves.json", curves)
        if best:
            _, selection, report = best
            preserve(directory / "validation.json", report)
            selection["validation_sha256"] = digest(directory / "validation.json")
            preserve(directory / "selection.json", selection)
            selections.append(selection)
        else:
            preserve(
                directory / "selection.json", {"selected": False, "test_used_for_selection": False}
            )
    if plan["baseline_hashes"] != hashes() or plan["code_sha256"] != contracts():
        raise ValueError("V29 fitting changed during training")
    winner = (
        max(
            selections,
            key=lambda s: (s["false_notes_removed"], s["margin_false_notes_removed"], -s["epoch"]),
        )
        if selections
        else None
    )
    result = {
        "selected": winner is not None,
        "winner": winner,
        "plan_sha256": digest(output / "plan.json"),
        "test_used_for_selection": False,
        "no_user_audio_or_scores": True,
        "new_weights_deployed": False,
    }
    preserve(output / "batch-selection.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("mixtures", type=Path)
    parser.add_argument("warm", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.baseline, args.mixtures, args.warm, args.output)
