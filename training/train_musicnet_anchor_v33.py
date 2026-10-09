"""Fit a new real-piano correction while freezing the complete released V32 model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from current_bass_v32_baseline import hashes
from musicnet_anchor_model import (
    VERSION,
    WARM_SHA,
    initialize,
    model,
    normalizer,
    probabilities,
)
from musicnet_v32_data import load
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import choose
from train_attack_release_v25 import supervised as acoustic_supervised

STAGES = (20, 40, 60, 80)
THRESHOLDS = (0.8, 0.85, 0.9, 0.95, 0.99)
GUARDIANS = (0.15, 0.3, 0.5)
PROFILES = (
    {
        "name": "real-piano-conservative",
        "width": 64,
        "cap": 2.0,
        "positive_weight": 16.0,
        "fragment_weight": 8.0,
        "seed": 278001,
    },
    {
        "name": "real-piano-protected",
        "width": 96,
        "cap": 4.0,
        "positive_weight": 32.0,
        "fragment_weight": 16.0,
        "seed": 278002,
    },
)


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/train_musicnet_anchor_v33.py",
        "training/musicnet_v32_data.py",
        "training/bass_v32_data.py",
        "training/bass_v32_regression_data.py",
        "training/reserved_musicnet_v33.py",
        "training/current_bass_v32_baseline.py",
        "training/prepare_bass_v32_baseline.py",
        "training/prepare_bass_v32_regression.py",
        "training/prepare_musicnet_v32.py",
        "training/acquire_musicnet_subset.py",
        "training/teacher_embedding_model.py",
        "training/fusion_bass_data.py",
        "training/fragment_support_model.py",
        "training/prepare_groove_bass.py",
        "training/acquire_groove_subset.py",
        "training/release_neighbor_features.py",
        "training/musicnet_anchor_model.py",
        "training/attack_release_model.py",
        "training/fusion_model.py",
        "training/pitch_interval_coverage.py",
        "training/periodicity_features.py",
        "training/score_declutter_local.py",
        "training/score_early_consensus.py",
        "training/bass_joint_timing.py",
        "training/partial_hold_training.py",
        "training/evaluate_musicnet_anchor_v33.py",
        "training/fresh_musicnet_anchor_v33.py",
        "training/train_attack_release_v25.py",
        "training/train_fusion_v27.py",
    )
    return {name: digest(root / name) for name in names}


def supervised(items, profile):
    x, frames, attacks, endings, labels, weights, counts = acoustic_supervised(
        items, profile["positive_weight"], profile["fragment_weight"]
    )
    phases = []
    for corpus in sorted({i["corpus"] for i in items}):
        group = [
            i
            for i in items
            if i["corpus"] == corpus and np.any(i["mask"] & i["eligible"])
        ]
        phases.extend(
            np.column_stack((item["periodicity"], item["release_neighbors"]))[
                item["mask"] & item["eligible"]
            ]
            for item in group
        )
    phase = np.concatenate(phases)
    if len(phase) != len(x):
        raise ValueError("Acoustic/recurrence supervision misaligned")
    return x, frames, attacks, endings, phase, labels, weights, counts


def train(folder, piano, regressions, warm, output):
    if digest(warm) != WARM_SHA:
        raise ValueError("Changed released V32 warm-start checkpoint")
    raw = load(folder, piano)
    training = [i for i in raw if i["group"] == "train"]
    validation = [i for i in raw if i["group"] == "validation"]
    piano_plan = json.loads((piano / "plan.json").read_text())
    expected = {
        "train": 702 + piano_plan["expected_counts"]["train"],
        "validation": 206 + piano_plan["expected_counts"]["validation"],
    }
    if len(training) != expected["train"] or len(validation) != expected["validation"]:
        raise ValueError("Incorrect V33 fitting partition")
    from reserved_musicnet_v33 import declaration

    fresh_declaration = declaration(piano)
    regression_manifest = json.loads((regressions / "manifest.json").read_text())
    if len(regression_manifest["items"]) != 469 or any(
        i["group"] != "test" for i in regression_manifest["items"]
    ):
        raise ValueError("Incomplete consumed regression identities")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "baseline_root": str(folder),
        "piano_root": str(piano),
        "piano_manifest_sha256": digest(piano / "manifest.json"),
        "piano_plan_sha256": digest(piano / "plan.json"),
        "expected_fitting_counts": expected,
        "regression_root": str(regressions),
        "regression_manifest_sha256": digest(regressions / "manifest.json"),
        "regression_plan_sha256": digest(regressions / "plan.json"),
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
        "architecture": "Freeze the complete released V32 network and its 155-feature normalizer, including its learned correction. A new neutral 103-input branch learns from recurrence/release-neighbor evidence and the acoustic teacher's pre-logit spectral representation. Only the new branch learns; it shifts REMOVE by at most cap 2/4. No fitted probabilities, labels or source IDs as branch inputs.",
        "teacher_anchor": {
            "frozen_parameters": True,
            "teacher_dropout_disabled": True,
            "zero_initial_correction": True,
            "caps": [2.0, 4.0],
            "no_fitted_probabilities_or_labels_as_branch_inputs": True,
        },
        "required_consumed_regressions": 469,
        "fresh_policy": "Only after both frozen settings pass every consumed regression: all original acquired MusicNet reserved works, three predeclared 20-second excerpts per recording, plus 32 new authored fixtures with seeds 278101–278132. Reserved NSynth/GMD sources are now consumed evidence, not new first-pass data. Both settings must reduce false notes and preserve all attacks, holds, pitch coverage and fixed-pair timing.",
        "new_first_pass_seeds": list(range(278101, 278133)),
        "first_pass_declaration": fresh_declaration,
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
        mean, scale = normalizer(saved)
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
        network = initialize(model(profile["width"], profile["cap"]), saved)
        teacher_state = {k: v.clone() for k, v in network.anchor.state_dict().items()}
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=0.0003, weight_decay=0.002
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=80, eta_min=0.00003
        )
        preserve(
            directory / "run.json",
            {
                "profile": profile,
                "plan_sha256": digest(output / "plan.json"),
                "training_counts": counts,
                "training_events": len(y),
                "training_recordings": expected["train"],
                "validation_recordings": expected["validation"],
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
                        network(
                            tf[indices],
                            ta[indices],
                            te[indices],
                            tx[indices],
                            tp[indices],
                        ),
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
                if any(
                    not torch.equal(v, teacher_state[k])
                    for k, v in network.anchor.state_dict().items()
                ):
                    raise ValueError("Frozen teacher changed during fitting")
                ps = [probabilities(network, i, (mean, scale)) for i in validation]
                chosen, search = choose(
                    validation, ps, guards, THRESHOLDS, GUARDIANS, memo
                )
                preserve(directory / f"search-{epoch:03d}.json", search)
                checkpoint = directory / f"epoch-{epoch:03d}.pt"
                torch.save(
                    {
                        "version": VERSION,
                        "width": profile["width"],
                        "cap": profile["cap"],
                        "epoch": epoch,
                        "state_dict": network.state_dict(),
                        "mean": torch.from_numpy(mean),
                        "scale": torch.from_numpy(scale),
                    },
                    checkpoint,
                )
                row.update(
                    selected=chosen is not None,
                    false_notes_removed=chosen["raw"]["false_notes_removed"]
                    if chosen
                    else 0,
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
                                "cap": profile["cap"],
                                "epoch": epoch,
                                "checkpoint": str(checkpoint.relative_to(output)),
                                "checkpoint_sha256": digest(checkpoint),
                                "threshold": chosen["raw"]["remove_threshold"],
                                "guardian_threshold": chosen["raw"][
                                    "guardian_threshold"
                                ],
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
                directory / "selection.json",
                {"selected": False, "test_used_for_selection": False},
            )
    if (
        plan["baseline_hashes"] != hashes()
        or plan["code_sha256"] != contracts()
        or plan["baseline_manifest_sha256"] != digest(folder / "manifest.json")
        or plan["piano_manifest_sha256"] != digest(piano / "manifest.json")
        or plan["warm_start_sha256"] != digest(warm)
        or plan["first_pass_declaration"] != declaration(piano)
    ):
        raise ValueError("V33 fitting changed during training")
    winner = (
        max(
            selections,
            key=lambda s: (
                s["false_notes_removed"],
                s["margin_false_notes_removed"],
                -s["epoch"],
            ),
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
    parser.add_argument("piano", type=Path)
    parser.add_argument("regressions", type=Path)
    parser.add_argument("warm", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.baseline, args.piano, args.regressions, args.warm, args.output)
