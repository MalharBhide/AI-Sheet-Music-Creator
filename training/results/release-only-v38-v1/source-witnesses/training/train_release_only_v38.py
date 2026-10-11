"""Fit a release-only specialist on old corpora plus new public-domain Joplin clocks."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from chord_boundary_v36 import context
from current_bass_v32_baseline import hashes
from musicnet_v32_data import load
from prepare_mutopia_boundaries import load as new_items
from prepare_robust_training_stems import digest, preserve
from release_only_v38 import VERSION, labels, model, probabilities
from score_chord_boundary_v36 import evaluate

PROFILES = (
    {"name": "release-small", "width": 16, "seed": 283001, "keep_weight": 1, "stay_weight": 4},
    {"name": "release-wide", "width": 24, "seed": 283002, "keep_weight": 1, "stay_weight": 8},
)
STAGES = (10, 20, 40)
TIMING = (.8, .9, .95, .99, .995, .999)
REMOVAL = (1.,)


def contracts(extra_sources=()):
    root = Path(__file__).resolve().parents[1]
    names = ("chord_boundary_v36.py", "score_chord_boundary_v36.py", "train_chord_boundary_v36.py",
             "musicnet_v32_data.py", "release_only_v38.py", "train_release_only_v38.py",
             "prepare_mutopia_boundaries.py", "acquire_mutopia_boundaries.py", "protected_boundary_targets_v37.py", "bass_v32_data.py", "score_bass_joint_timing.py",
             "bass_joint_timing.py", "pitch_interval_coverage.py")
    return {"training/"+name: digest(root / "training" / name) for name in (*names, *extra_sources)}


def tensors(items, labeler=labels):
    corpora = Counter(i["corpus"] for i in items)
    chunks = {key: [] for key in ("x", "attack", "release", "y", "weight")}
    for item in items:
        y = labeler(item)
        chosen = np.any(y != -100, axis=1)
        if not chosen.any():
            continue
        chunks["x"].append(context(item)[chosen])
        chunks["attack"].append(item["attack_frames"][chosen])
        chunks["release"].append(item["release_frames"][chosen])
        chunks["y"].append(y[chosen])
        chunks["weight"].append(np.full(chosen.sum(), 1/(chosen.sum()*corpora[item["corpus"]])))
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    result["weight"] /= result["weight"].mean()
    return result


def fit(baseline, piano, new_data, output, *, labeler=labels, profiles=PROFILES, version=VERSION,
        extra_sources=(), target_policy="Unique reference-boundary targets within the observed local window."):
    # Existing readers verify waveform/cache hashes and composition-disjoint
    # train/validation groups, and refuse test or incomplete source partitions.
    initial = hashes()
    items = load(baseline, piano)
    items.extend(new_items(new_data))
    training = [i for i in items if i["group"] == "train"]
    validation = [i for i in items if i["group"] == "validation"]
    if len(training) != 781 or len(validation) != 225:
        raise ValueError("Changed declared V38 fitting partition")
    output.mkdir(exist_ok=False)
    plan = {"version": version, "baseline_hashes": initial, "source_sha256": contracts(extra_sources),
            "baseline_manifest_sha256": digest(baseline / "manifest.json"),
            "piano_manifest_sha256": digest(piano / "manifest.json"),
            "baseline_root": str(baseline), "piano_root": str(piano), "new_root": str(new_data),
            "new_manifest_sha256": digest(new_data / "manifest.json"),
            "regression_root": "/training/bass-v32-regression-v1",
            "regression_manifest_sha256": digest(Path("/training/bass-v32-regression-v1/manifest.json")),
            "additional_consumed_root": "/training/musicnet-full-v34-first-pass-v1",
            "additional_consumed_manifest_sha256": digest(Path("/training/musicnet-full-v34-first-pass-v1/manifest.json")),
            "training_recordings": len(training), "validation_recordings": len(validation),
            "profiles": profiles, "epochs": 40, "updates_per_epoch": 25, "batch_size": 256,
            "learning_rate": .0005, "weight_decay": .003, "stages": STAGES,
            "timing_thresholds": TIMING, "removal_thresholds": REMOVAL,
            "max_boundary_shift_seconds": .08, "guardian_threshold": .15,
            "architecture": "Separate local attack/release CNNs with 84 observed context, eight same-key and eight chord interval features. Release-only trainable head; immutable KEEP/attack heads. No shared chord ending, new notes or pitch rewriting.",
            "supervision": "Conservative pitch-support labels. Unique same-pitch attack matches within 200 ms provide boundary targets only within 90 ms. Corpus/recording-balanced losses; positive KEEP weighting; boundary class balancing fitted on training only. References, source IDs and fitted model scores are never inference inputs.",
            "boundary_target_policy": target_policy,
            "release_policy": "Freeze one validation winner. All 519 consumed regressions must pass at normal and stricter settings before any new first pass; preserve attacks, holds, supplied pitch intervals, per-recording fixed-pair onset/release error, repeated spacing, chord onset spacing, unsupported hold duration and false-positive counts. Deletion earns no timing credit. Require positive release gain at both validation settings, and nonnegative onset and false-note gain. Every preservation gate remains; regression and fresh passes require nonnegative release gain. This is a new specialist objective declared before training, not a reuse of V36/V37 weights or outcomes. No runner-up or gate relaxation after regression.",
            "fresh_policy": "Only after consumed gates pass: 40 new project-authored MP3 clips, seeds 283101-283140, plus four newly rendered reserved Mutopia clips, alternating original/texture generators, 48/128/192 kbps cycling. Both settings must pass the same gates. Runtime feature/export parity and native integration required before deployment.",
            "no_test_used_for_selection": True, "no_user_audio_or_scores": True,
            "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output / "plan.json", plan)
    target_cache = {i["id"]: labeler(i) for i in items}

    def cached_labels(item):
        return target_cache[item["id"]]

    data = tensors(training, cached_labels)
    mean, scale = data["x"].mean(axis=0), np.maximum(data["x"].std(axis=0), .01)
    x, attacks, releases = [torch.from_numpy(v.astype(np.float32)) for v in (
        (data["x"]-mean)/scale, data["attack"], data["release"])]
    y, weights = torch.from_numpy(data["y"]), torch.from_numpy(data["weight"].astype(np.float32))
    counts = [np.bincount(data["y"][:, head][data["y"][:, head] >= 0], minlength=size)
              for head, size in enumerate((2, 7, 7))]
    preserve(output / "supervision.json", {"events": len(y), "class_counts": [c.tolist() for c in counts],
                                          "corpora": dict(Counter(i["corpus"] for i in training)),
                                          "validation_never_used_for_normalization": True})
    guardian = BassHarmonic().models[1]
    guards = [guardian.probability(i["base_x"]) for i in validation]
    torch.set_num_threads(2)
    candidates = []
    for profile in profiles:
        directory = output / profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(profile["width"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=.00005)
        class_weights = [torch.tensor([profile["keep_weight"], 1.], dtype=torch.float32)]
        for c in counts[1:]:
            balanced = np.clip(np.sqrt(c.sum()/np.maximum(c, 1)), 1, 8)
            class_weights.append(torch.from_numpy(balanced.astype(np.float32)))
        class_weights[2][3] *= profile["stay_weight"]
        curves, best, memo = [], None, {}
        for epoch in range(1, 41):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(25):
                idx = torch.from_numpy(rng.integers(0, len(y), 256))
                heads = network(attacks[idx], releases[idx], x[idx])
                loss = torch.tensor(0.)
                for head, logits in enumerate(heads):
                    available = y[idx, head] != -100
                    if available.any():
                        terms = torch.nn.functional.cross_entropy(logits, y[idx, head],
                            weight=class_weights[head], ignore_index=-100, reduction="none")
                        loss = loss+(terms[available]*weights[idx][available]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite boundary loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            curve = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic()-started}
            if epoch in STAGES:
                ps = [probabilities(network, i, (mean, scale)) for i in validation]
                # Log classification separately from the musical release gate.
                # A better loss or majority-class accuracy cannot authorize shipping.
                classification = {}
                for head, name in enumerate(("support", "onset", "release")):
                    truth = np.concatenate([cached_labels(i)[:, head] for i in validation])
                    predicted = np.concatenate([p[head].argmax(axis=1) for p in ps])
                    use = truth != -100
                    size = 2 if head == 0 else 7
                    matrix = np.zeros((size, size), np.int64)
                    np.add.at(matrix, (truth[use], predicted[use]), 1)
                    classification[name] = {"labeled_events": int(use.sum()),
                        "accuracy": float(np.mean(predicted[use] == truth[use])) if use.any() else None,
                        "confusion_matrix": matrix.tolist()}
                preserve(directory / f"classification-{epoch:03d}.json", classification)
                checkpoint = directory / f"epoch-{epoch:03d}.pt"
                torch.save({"version": version, "width": profile["width"], "epoch": epoch,
                            "state_dict": network.state_dict(), "mean": torch.from_numpy(mean),
                            "scale": torch.from_numpy(scale)}, checkpoint)
                search, reports = [], []
                for timing in TIMING:
                    for removal in REMOVAL:
                        raw = evaluate(validation, ps, guards, timing, removal, memo)
                        strict = evaluate(validation, ps, guards, min(.9999, timing+.03), min(1., removal+.005), memo)
                        allowed = all(r["passes"] and r["onset_gain_seconds"] >= -1e-9
                                      and r["release_gain_seconds"] > 1e-6 and r["false_notes_removed"] >= 0
                                      for r in (raw, strict))
                        search.append({"eligible": allowed, "raw": {k: v for k, v in raw.items() if k != "per_recording"},
                                       "strict": {k: v for k, v in strict.items() if k != "per_recording"}})
                        reports.append((raw, strict))
                        if allowed:
                            key = (raw["onset_gain_seconds"]+raw["release_gain_seconds"], raw["false_notes_removed"], -epoch)
                            if best is None or key > best[0]:
                                best = (key, {"profile": profile, "epoch": epoch,
                                    "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": digest(checkpoint),
                                    "timing_threshold": timing, "remove_threshold": removal,
                                    "plan_sha256": digest(output / "plan.json")}, raw, strict)
                preserve(directory / f"search-{epoch:03d}.json", search)
                # Retain the most active fixed setting for diagnosis only; never
                # substitute it for a candidate that passed every release gate.
                diagnostic = max(reports, key=lambda pair: pair[0]["retimed_notes"])
                preserve(directory / f"diagnostic-{epoch:03d}.json", {"raw": diagnostic[0], "strict": diagnostic[1]})
                curve["selected"] = best is not None
            curves.append(curve)
            print(json.dumps({"profile": profile["name"], **curve}), flush=True)
        preserve(directory / "learning-curves.json", curves)
        if best is None:
            preserve(directory / "selection.json", {"selected": False, "test_used_for_selection": False})
        else:
            _, selected, raw, strict = best
            preserve(directory / "validation.json", {"raw": raw, "strict": strict})
            selected["validation_sha256"] = digest(directory / "validation.json")
            preserve(directory / "selection.json", selected)
            candidates.append(best)
    if initial != hashes() or plan["source_sha256"] != contracts(extra_sources):
        raise ValueError("Production baseline or training code changed during fitting")
    winner = max(candidates, key=lambda c: c[0])[1] if candidates else None
    preserve(output / "batch-selection.json", {"selected": winner is not None, "winner": winner,
        "plan_sha256": digest(output / "plan.json"), "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False})
    print(json.dumps({"complete": True, "winner": winner}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "piano", "new_data", "output"):
        parser.add_argument(name, type=Path)
    fit(**vars(parser.parse_args()))
