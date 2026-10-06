"""Train recurrence-based decluttering on V25 survivors plus new NSynth instruments."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from current_bass_v25_baseline import hashes
from partial_hold_training import support_boost
from periodicity_bass_data import load
from periodicity_model import VERSION, inputs, model, probabilities
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import choose

STAGES = (20, 40, 60, 80)
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
GUARDIANS = (0.15, 0.3, 0.5)
PROFILES = (
    {
        "name": "recurrence-conservative",
        "width": 64,
        "positive_weight": 16.0,
        "fragment_weight": 8.0,
        "seed": 271001,
    },
    {
        "name": "recurrence-protected",
        "width": 96,
        "positive_weight": 32.0,
        "fragment_weight": 16.0,
        "seed": 271002,
    },
)


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/train_periodicity_v26.py",
        "training/periodicity_bass_data.py",
        "training/periodicity_model.py",
        "training/periodicity_features.py",
        "training/score_declutter_local.py",
        "training/score_early_consensus.py",
        "training/bass_joint_timing.py",
        "training/partial_hold_training.py",
        "training/evaluate_periodicity_v26.py",
    )
    return {name: digest(root / name) for name in names}


def supervised(items, profile):
    values, labels, weights, counts = [], [], [], {}
    corpora = sorted({i["corpus"] for i in items})
    for corpus in corpora:
        group = [i for i in items if i["corpus"] == corpus and np.any(i["mask"] & i["eligible"])]
        counts[corpus] = {"recordings": len(group), "classes": [0, 0]}
        for item in group:
            select = item["mask"] & item["eligible"]
            y = (1 - item["y"][select]).astype(np.int64)
            values.append(inputs(item)[select])
            labels.append(y)
            boost = support_boost(item, profile["fragment_weight"])[select]
            weights.append(
                np.where(y == 0, profile["positive_weight"], 1.0)
                * boost
                / (len(corpora) * len(group) * len(y))
            )
            counts[corpus]["classes"] = (
                np.asarray(counts[corpus]["classes"]) + np.bincount(y, minlength=2)
            ).tolist()
    x, y, w = map(np.concatenate, (values, labels, weights))
    return x, y, w / w.mean(), counts


def train(folder, nsynth, output):
    raw = load(folder, nsynth)
    training = [i for i in raw if i["group"] == "train"]
    validation = [i for i in raw if i["group"] == "validation"]
    if len(training) != 639 or len(validation) != 178:
        raise ValueError("Incorrect V26 fitting partition")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "baseline_root": str(folder),
        "baseline_manifest_sha256": digest(folder / "manifest.json"),
        "nsynth_root": str(nsynth),
        "nsynth_manifest_sha256": digest(nsynth / "manifest.json"),
        "profiles": list(PROFILES),
        "stages": list(STAGES),
        "updates_per_epoch": 40,
        "batch_size": 512,
        "learning_rate": 0.001,
        "minimum_learning_rate": 0.0001,
        "weight_decay": 0.002,
        "thresholds": list(THRESHOLDS),
        "guardian_thresholds": list(GUARDIANS),
        "architecture": "84 acoustic context + 63 normalized waveform recurrence observations, two-layer MLP KEEP/REMOVE",
        "required_consumed_regressions": 392,
        "fresh_policy": "Only after both frozen confidence settings pass every consumed regression. All four reserved NSynth instruments, six declared MP3 sequences each, plus 32 new project-authored clips with seeds 271101–271132. Both settings must reduce false notes and preserve all attacks, holds, pitch coverage and fixed-pair timing.",
        "new_first_pass_seeds": list(range(271101, 271133)),
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
        x, y, w, counts = supervised(training, profile)
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 0.01)
        tx, ty, tw = [
            torch.from_numpy(a)
            for a in (((x - mean) / scale).astype(np.float32), y, w.astype(np.float32))
        ]
        network = model(profile["width"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.0001)
        preserve(
            directory / "run.json",
            {
                "profile": profile,
                "plan_sha256": digest(output / "plan.json"),
                "training_counts": counts,
                "training_events": len(y),
                "training_recordings": 639,
                "validation_recordings": 178,
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
                indices = torch.from_numpy(rng.integers(0, len(y), 512))
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    torch.nn.functional.cross_entropy(
                        network(tx[indices]), ty[indices], reduction="none"
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
                ps = [probabilities(network, inputs(i), (mean, scale)) for i in validation]
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
        raise ValueError("V26 fitting changed during training")
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
    parser.add_argument("nsynth", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.baseline, args.nsynth, args.output)
