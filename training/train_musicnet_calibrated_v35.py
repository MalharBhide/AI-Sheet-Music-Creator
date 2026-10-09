"""Fit new real-piano protection and bounded rejection heads using no test labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from consumed_v34_identity import declaration, first_pass_declaration
from current_bass_v32_baseline import hashes
from musicnet_calibrated_guardian_model import (
    VERSION,
    WARM_SHA,
    initialize,
    model,
    normalizer,
    probabilities,
)
from musicnet_v32_data import load
from prepare_robust_training_stems import digest, preserve
from score_musicnet_calibrated import choose
from train_musicnet_full_v34 import contracts as prior_contracts
from train_musicnet_full_v34 import supervised

STAGES = (20, 40, 60, 80)
THRESHOLDS = (0.8, 0.85, 0.9, 0.95, 0.99)
GUARDIANS = (0.15, 0.3, 0.5)
PROFILES = (
    {
        "name": "calibrated-conservative",
        "width": 128,
        "cap": 4.0,
        "positive_weight": 16.0,
        "fragment_weight": 8.0,
        "seed": 280001,
    },
    {
        "name": "calibrated-protected",
        "width": 192,
        "cap": 6.0,
        "positive_weight": 32.0,
        "fragment_weight": 16.0,
        "seed": 280002,
    },
)


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/train_musicnet_calibrated_v35.py",
        "training/musicnet_calibrated_guardian_model.py",
        "training/consumed_v34_identity.py",
        "training/score_musicnet_calibrated.py",
        "training/oxford_midi_test_source.py",
        "training/acquire_oxford_midi_test.py",
    )
    return {**prior_contracts(), **{name: digest(root / name) for name in names}}


def train(baseline, piano, regressions, consumed_first_pass, oxford, warm, output):
    if digest(warm) != WARM_SHA:
        raise ValueError("Changed released V32 checkpoint")
    items = load(baseline, piano)
    fitting = [i for i in items if i["group"] == "train"]
    validation = [i for i in items if i["group"] == "validation"]
    if len(fitting) != 765 or len(validation) != 221:
        raise ValueError("Changed fitting partition")
    consumed = declaration(regressions, consumed_first_pass)
    fresh = first_pass_declaration(oxford)
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "baseline_root": str(baseline),
        "piano_root": str(piano),
        "baseline_manifest_sha256": digest(baseline / "manifest.json"),
        "piano_manifest_sha256": digest(piano / "manifest.json"),
        "piano_plan_sha256": digest(piano / "plan.json"),
        "consumed_regression_declaration": consumed,
        "required_consumed_regressions": 519,
        "first_pass_declaration": fresh,
        "profiles": list(PROFILES),
        "stages": list(STAGES),
        "warm_start": str(warm),
        "warm_start_sha256": WARM_SHA,
        "thresholds": list(THRESHOLDS),
        "guardian_thresholds": list(GUARDIANS),
        "expected_fitting_counts": {"train": 765, "validation": 221},
        "updates_per_epoch": 40,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "minimum_learning_rate": 0.00003,
        "weight_decay": 0.002,
        "architecture": "Complete frozen V32 anchor with its 155-feature normalizer. Independent 955-input removal correction and acoustic protection branches use full frozen spectral encodings, normalized context, recurrence and neighbors. Removal correction bounded by cap 4/6. Protection starts at .99 and learns independently; neither head consumes source IDs, reference labels or fitted probabilities as features. Preceding nine released stages remain unchanged.",
        "supervision": "Same conservative pitch/attack/hold labels, exclusion of ambiguous negatives, corpus/recording-balanced loss and protected partial holds. Equal average of REMOVE cross entropy and KEEP protection BCE using identical per-event weights. No regression or reserved examples fit either head.",
        "selection": "Both confidence settings must preserve every validation recording and independently remove at least one real-piano false note. Rank real gain, stricter real gain, overall gains, earlier epoch and fixed profile order. No authored-only winner.",
        "release_policy": "Freeze one validation winner. Require all 519 consumed recording gates before the single 40-record first pass, separately positive Oxford real-piano benefit at both settings, exact portable masks, physical-feature parity and native integration. No retuning, runner-up, replacement recordings or relaxed coverage/timing gates.",
        "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True,
    }
    output.mkdir(exist_ok=False)
    preserve(output / "plan.json", plan)
    torch.set_num_threads(2)
    saved = torch.load(warm, map_location="cpu", weights_only=True)
    mean, scale = normalizer(saved)
    selections = []
    for profile in PROFILES:
        directory = output / profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        x, f, a, e, phase, y, w, counts = supervised(fitting, profile)
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
        original = {k: v.clone() for k, v in network.anchor.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [p for p in network.parameters() if p.requires_grad], lr=0.0003, weight_decay=0.002
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.00003)
        preserve(
            directory / "run.json",
            {
                "profile": profile,
                "plan_sha256": digest(output / "plan.json"),
                "training_counts": counts,
                "training_events": len(y),
                "training_recordings": 765,
                "validation_recordings": 221,
                "test_used_for_selection": False,
            },
        )
        best, curves, memo = None, [], {}
        for epoch in range(1, 81):
            network.train()
            started, losses = time.monotonic(), []
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(y), 256))
                optimizer.zero_grad(set_to_none=True)
                logits, protection = network.both(tf[idx], ta[idx], te[idx], tx[idx], tp[idx])
                rejection = torch.nn.functional.cross_entropy(logits, ty[idx], reduction="none")
                support = torch.nn.functional.binary_cross_entropy_with_logits(
                    protection, (1 - ty[idx]).float(), reduction="none"
                )
                loss = ((rejection + support) * 0.5 * tw[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite calibrated protection loss")
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
                    not torch.equal(v, original[k]) for k, v in network.anchor.state_dict().items()
                ):
                    raise ValueError("Released anchor changed")
                outputs = [probabilities(network, i, (mean, scale)) for i in validation]
                ps, guards = [r[0] for r in outputs], [r[1] for r in outputs]
                chosen, search = choose(validation, ps, guards, THRESHOLDS, GUARDIANS, memo)
                preserve(directory / f"search-{epoch:03d}.json", search)
                path = directory / f"epoch-{epoch:03d}.pt"
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
                    path,
                )
                row["selected"] = chosen is not None
                if chosen:
                    key = (
                        chosen["real_piano_gain"],
                        chosen["margin_real_piano_gain"],
                        chosen["raw"]["false_notes_removed"],
                        chosen["margin"]["false_notes_removed"],
                        -epoch,
                    )
                    selection = {
                        "profile": profile["name"],
                        "width": profile["width"],
                        "cap": profile["cap"],
                        "epoch": epoch,
                        "checkpoint": str(path.relative_to(output)),
                        "checkpoint_sha256": digest(path),
                        "threshold": chosen["raw"]["remove_threshold"],
                        "guardian_threshold": chosen["raw"]["guardian_threshold"],
                        "real_piano_gain": key[0],
                        "margin_real_piano_gain": key[1],
                        "false_notes_removed": key[2],
                        "margin_false_notes_removed": key[3],
                        "plan_sha256": digest(output / "plan.json"),
                        "test_used_for_selection": False,
                    }
                    row.update(
                        {
                            k: selection[k]
                            for k in (
                                "real_piano_gain",
                                "margin_real_piano_gain",
                                "false_notes_removed",
                            )
                        }
                    )
                    if best is None or key > best[0]:
                        best = key, selection, chosen
            curves.append(row)
            print(json.dumps({"profile": profile["name"], **row}), flush=True)
        preserve(directory / "learning-curves.json", curves)
        if best:
            _, selected, report = best
            preserve(directory / "validation.json", report)
            selected["validation_sha256"] = digest(directory / "validation.json")
            preserve(directory / "selection.json", selected)
            selections.append(selected)
        else:
            preserve(
                directory / "selection.json", {"selected": False, "test_used_for_selection": False}
            )
    if (
        plan["code_sha256"] != contracts()
        or plan["baseline_hashes"] != hashes()
        or plan["consumed_regression_declaration"] != declaration(regressions, consumed_first_pass)
        or plan["first_pass_declaration"] != first_pass_declaration(oxford)
        or plan["baseline_manifest_sha256"] != digest(baseline / "manifest.json")
        or plan["piano_manifest_sha256"] != digest(piano / "manifest.json")
        or plan["warm_start_sha256"] != digest(warm)
    ):
        raise ValueError("Calibrated training contract changed during fitting")
    winner = (
        max(
            selections,
            key=lambda s: (
                s["real_piano_gain"],
                s["margin_real_piano_gain"],
                s["false_notes_removed"],
                s["margin_false_notes_removed"],
                -s["epoch"],
            ),
        )
        if selections
        else None
    )
    preserve(
        output / "batch-selection.json",
        {
            "selected": winner is not None,
            "winner": winner,
            "plan_sha256": digest(output / "plan.json"),
            "test_used_for_selection": False,
            "no_user_audio_or_scores": True,
            "new_weights_deployed": False,
        },
    )
    print(json.dumps({"completed": True, "winner": winner}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "piano",
        "regressions",
        "consumed_first_pass",
        "oxford",
        "warm",
        "output",
    ):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    train(**vars(args))
