"""Fit new dual-head left-hand models with independent authored note evidence."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from app.services.chord_release import ReleaseModel
from current_bass_v32_baseline import hashes
from left_precision_v41 import VERSION, model, probabilities
from musicnet_v32_data import load
from prepare_left_support_v41 import load as authored_items
from prepare_robust_training_stems import digest, preserve
from score_left_precision_v39 import baseline, evaluate
from train_left_precision_v40 import SOURCES as PRIOR_SOURCES
from train_left_precision_v40 import deployed_contract, tensors

PROFILES = (
    {"name": "dual-small", "width": 12, "keep_weight": 8, "guard_keep_weight": 32, "seed": 286001},
    {"name": "dual-protected", "width": 20, "keep_weight": 16, "guard_keep_weight": 64, "seed": 286002},
)
THRESHOLDS = (.9, .95, .98, .99, .995, .999)
STAGES = (25, 50, 100)
FOLDERS = {"baseline": Path("/training/bass-v32-baseline-v1"),
    "piano": Path("/training/musicnet-v32-observations-v1"),
    "authored": Path("/training/left-support-v41-observations")}
SOURCES = tuple(dict.fromkeys((*PRIOR_SOURCES, "left_precision_v41.py", "train_left_precision_v41.py",
    "evaluate_left_precision_v41.py", "evaluate_left_precision_v40.py", "prepare_left_support_v41.py",
    "prepare_nsynth_bass.py", "acquire_nsynth_subset.py", "prepare_slakh_bass.py", "prepare_slakh_training.py",
    "attack_local_features.py", "release_local_features.py", "release_neighbor_features.py", "fresh_left_precision_v41.py",
    "prepare_bass_positive_data.py", "prepare_bass_texture_v12.py")))


def train(production, output):
    torch.set_num_threads(2)
    current, historical = deployed_contract(production), hashes()
    prior = Path("/training/left-precision-v40-v1/consumed-regression.json")
    if json.loads(prior.read_text())["passes"]:
        raise ValueError("This new study requires completed, withheld V40 evidence")
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCES}
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "production": current, "historical_v32": historical,
        "prior_rejected_regression_sha256": digest(prior), "source_sha256": sources,
        "fitting_manifests": {k: digest(v/"manifest.json") for k, v in FOLDERS.items()},
        "training_recordings": 853, "validation_recordings": 245, "profiles": PROFILES,
        "epochs": 100, "updates_per_epoch": 40, "batch_size": 256, "stages": STAGES,
        "learning_rate": .0005, "minimum_learning_rate": .00005, "weight_decay": .003,
        "thresholds": THRESHOLDS, "stricter_margin": .005,
        "architecture": "Fresh V40 spectral/periodic CNN rejection head plus independent physical KEEP MLP. Guardian uses normalized 163 context/recurrence features and per-channel mean/max/std of all three windows. Removal probability is min of both heads; neither head inherits weights, logits, reference labels, pitch identity, filename or source IDs.",
        "supervision": "All supplied pitch fragments, genuine attacks and held matches remain KEEP. Train-only normalization and corpus/recording weights. Separate weighted cross-entropy for each independent head. New authored NSynth set has 88 train/24 validation clips: quiet bass, overlapping long notes, chords, short and variable repeats. Original instrument partitions preserved, all four reserved instruments untouched. Authored damping tails protected; no human-key-release claim for gated samples.",
        "composition": "Compute deletion probability from complete V32 input. Run current V38 releases on complete unchanged input; apply deletion afterward. No onset, release, pitch or velocity edits for retained notes.",
        "validation_gate": "Every recording must preserve attacks, holds, pitch support, fixed-pair onset/release timing, repeated spacing, chord timing and unsupported duration. Positive overall false-note reduction and separately positive MusicNet real-piano gain at normal and stricter thresholds. Freeze one validation winner without any test use.",
        "subsequent_gate": "All 563 consumed regressions at both thresholds, followed by one independent 40-clip original/texture MP3 first pass with seeds 286101-286140 cycling 48/128/192 kbps and separately positive gain. Never fit consumed failures, retune, choose a runner-up or replace a first-pass clip after failure. Export, complete physical/runtime parity and native functional tests before website changes.",
        "no_user_audio_or_scores": True, "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output/"plan.json", plan)
    print(json.dumps({"plan_frozen": True, "loading_fitting_only": True}), flush=True)
    items = load(FOLDERS["baseline"], FOLDERS["piano"])+authored_items(FOLDERS["authored"])
    training, validation = [[i for i in items if i["group"] == g] for g in ("train", "validation")]
    if (len(training) != 853 or len(validation) != 245 or len({i["id"] for i in items}) != len(items)
            or {i["source_group"] for i in training} & {i["source_group"] for i in validation}
            or sum(i["corpus"] == "musicnet-real-piano" for i in validation) != 15):
        raise ValueError("Incomplete or leaking dual-head fitting partitions")
    arrays, counts = tensors(training)
    coarse, attack, release, x, target, weights = arrays
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
    inputs = [torch.from_numpy(v.astype(np.float32)) for v in (coarse, attack, release, (x-mean)/scale)]
    target, weights = torch.from_numpy(target), torch.from_numpy(weights)
    preserve(output/"supervision.json", {"class_counts_keep_remove": np.bincount(target.numpy(), minlength=2).tolist(),
        "per_recording_class_counts": counts, "training_corpora": dict(Counter(i["corpus"] for i in training)),
        "train_only_normalization": True})
    with np.load(production/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        deployed = baseline(validation, ReleaseModel(saved))
    candidates = []
    for profile in PROFILES:
        directory = output/profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(profile["width"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=.00005)
        class_weights = [torch.tensor([profile[k], 1.], dtype=torch.float32)
                         for k in ("keep_weight", "guard_keep_weight")]
        curves, best, memo = [], None, {}
        for epoch in range(1, 101):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                heads = network(*[v[idx] for v in inputs])
                parts = [(torch.nn.functional.cross_entropy(logits, target[idx], weight=cw,
                          reduction="none")*weights[idx]).mean() for logits, cw in zip(heads, class_weights, strict=True)]
                loss = sum(parts)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite dual support loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append([float(part.detach()) for part in parts])
            scheduler.step()
            curve = {"epoch": epoch, "head_losses": np.mean(losses, axis=0).tolist(), "seconds": time.monotonic()-started}
            if epoch in STAGES:
                ps = [probabilities(network, item, (mean, scale)) for item in validation]
                checkpoint = directory/f"epoch-{epoch:03d}.pt"
                torch.save({"version": VERSION, "width": profile["width"], "epoch": epoch,
                    "state_dict": network.state_dict(), "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)}, checkpoint)
                search, reports = [], []
                for threshold in THRESHOLDS:
                    raw = evaluate(validation, deployed, ps, threshold, memo)
                    strict = evaluate(validation, deployed, ps, min(.9999, threshold+.005), memo)
                    allowed = all(r["passes"] and r["false_notes_removed"] > 0 and r["real_piano_gain"] > 0 for r in (raw, strict))
                    search.append({"eligible": allowed, "raw": {k: v for k, v in raw.items() if k != "per_recording"},
                        "strict": {k: v for k, v in strict.items() if k != "per_recording"}})
                    reports.append((raw, strict))
                    if allowed:
                        key = (raw["real_piano_gain"], strict["real_piano_gain"], raw["false_notes_removed"],
                               strict["false_notes_removed"], -epoch, threshold)
                        if best is None or key > best[0]:
                            best = (key, {"profile": profile, "epoch": epoch, "threshold": threshold,
                                "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": digest(checkpoint),
                                "plan_sha256": digest(output/"plan.json")}, raw, strict)
                preserve(directory/f"search-{epoch:03d}.json", search)
                diagnostic = max(reports, key=lambda pair: pair[0]["removed_notes"])
                preserve(directory/f"diagnostic-{epoch:03d}.json", {"raw": diagnostic[0], "strict": diagnostic[1]})
                curve["selected"] = best is not None
                curve["safest_setting"] = search[-1]
            curves.append(curve)
            print(json.dumps({"profile": profile["name"], **curve}), flush=True)
        preserve(directory/"learning-curves.json", curves)
        if best is None:
            preserve(directory/"selection.json", {"selected": False, "no_test_used_for_selection": True})
        else:
            _, chosen, raw, strict = best
            preserve(directory/"validation.json", {"raw": raw, "strict": strict})
            chosen["validation_sha256"] = digest(directory/"validation.json")
            preserve(directory/"selection.json", chosen)
            candidates.append(best)
    if (current != deployed_contract(production) or historical != hashes()
            or sources != {name: digest(root/name) for name in SOURCES}):
        raise ValueError("Production or fitting sources changed during training")
    winner = max(candidates, key=lambda v: v[0])[1] if candidates else None
    preserve(output/"batch-selection.json", {"selected": winner is not None, "winner": winner,
        "plan_sha256": digest(output/"plan.json"), "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False})
    print(json.dumps({"complete": True, "winner": winner}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    train(**vars(parser.parse_args()))
