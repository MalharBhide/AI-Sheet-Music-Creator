"""Train a left-hand rejection specialist; production remains unchanged unless gated."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from app.services.chord_release import ReleaseModel
from current_bass_v32_baseline import hashes
from left_precision_v39 import VERSION, model, observations, probabilities
from musicnet_v32_data import load
from prepare_mutopia_boundaries import load as mutopia
from prepare_robust_training_stems import digest, preserve
from score_left_precision_v39 import baseline, evaluate

PROFILES = (
    {"name": "support-small", "width": 12, "keep_weight": 16, "seed": 284001},
    {"name": "support-protected", "width": 20, "keep_weight": 64, "seed": 284002},
)
THRESHOLDS = (.8, .9, .95, .98, .99, .995, .999)
STAGES = (10, 20, 40)
SOURCES = ("left_precision_v39.py", "score_left_precision_v39.py", "train_left_precision_v39.py",
           "prepare_left_precision_replay.py", "chord_boundary_v36.py", "bass_joint_timing.py",
           "score_bass_joint_timing.py", "pitch_interval_coverage.py", "musicnet_v32_data.py",
           "bass_v32_data.py", "prepare_mutopia_boundaries.py")


def deployed_contract(production):
    path = production / "training/results/release-only-v38-v1/release-checks.json"
    release = json.loads(path.read_text())
    if (release["status"] != "verified" or not release["new_weights_deployed"]
            or not release["no_user_audio_or_scores"]
            or any(digest(production / name) != sha for name, sha in release["source_sha256"].items())
            or release["asset_sha256"] != digest(production / "backend/app/assets/chord-release-v1.npz")):
        raise ValueError("Production differs from the released V38 baseline")
    return {"release_checks_sha256": digest(path), "sources": release["source_sha256"],
            "release_asset_sha256": release["asset_sha256"]}


def tensors(items):
    corpora = Counter(i["corpus"] for i in items)
    chunks = [[] for _ in range(6)]
    counts = {}
    for item in items:
        chosen = item["eligible"] & item["mask"]
        if not chosen.any():
            continue
        inputs = observations(item)
        target = 1-item["y"][chosen].astype(np.int64)
        weight = np.full(chosen.sum(), 1/(chosen.sum()*corpora[item["corpus"]]), np.float32)
        for destination, source in zip(chunks, (*[v[chosen] for v in inputs], target, weight), strict=True):
            destination.append(source)
        counts[item["id"]] = np.bincount(target, minlength=2).tolist()
    arrays = [np.concatenate(chunk) for chunk in chunks]
    arrays[-1] /= arrays[-1].mean()
    return arrays, counts


def train(production, output):
    torch.set_num_threads(2)
    current = deployed_contract(production)
    historical = hashes()  # Strict V32 guard runs inside the isolated original reader.
    root = Path(__file__).resolve().parent
    sources = {name: digest(root / name) for name in SOURCES}
    folders = {"baseline": Path("/training/bass-v32-baseline-v1"),
               "piano": Path("/training/musicnet-v32-observations-v1"),
               "mutopia": Path("/training/mutopia-joplin-v38-observations")}
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "production": current, "historical_v32": historical,
        "source_sha256": sources, "fitting_manifests": {k: digest(v / "manifest.json") for k, v in folders.items()},
        "training_recordings": 781, "validation_recordings": 225, "profiles": PROFILES,
        "epochs": 40, "updates_per_epoch": 25, "batch_size": 256, "stages": STAGES,
        "learning_rate": .0005, "minimum_learning_rate": .00005, "weight_decay": .003,
        "thresholds": THRESHOLDS, "stricter_margin": .005,
        "architecture": "Fresh independent CNNs over coarse, attack and release pitch-relative windows; 100 physical context features. Two-class KEEP/REMOVE head. No inherited logits, absolute pitch, filenames, reference labels or source IDs as inference features.",
        "supervision": "Conservative KEEP for every supplied pitch fragment, attack and held match. Train-only corpus/recording-balanced weights and normalization; KEEP losses weighted 16 or 64.",
        "composition": "Compute removal permissions on complete V32 survivors; run deployed V38 release pass on the same complete survivors; only then apply permission. Retained clocks therefore remain bit-identical to deployed V38, independently of deletion context.",
        "validation_gate": "Preserve every recording's real attacks, holds and pitch coverage, fixed-pair onset/release error, repeated spacing, chord timing and unsupported duration. Positive false-note reduction and separately positive MusicNet real-piano reduction at normal and stricter settings. Freeze one validation winner; never select on consumed or fresh tests.",
        "subsequent_gate": "Before export require all 563 consumed recordings (519 V34-era plus 44 V38 first-pass), at both settings. Then one new independent original/texture MP3 first pass (40 seeds 284101-284140, cycling 48/128/192 kbps), separately positive benefit required. No runner-up or retuning after consumption. Portable/runtime parity and native pipeline tests before production changes.",
        "no_user_audio_or_scores": True, "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output / "plan.json", plan)
    print(json.dumps({"plan_frozen": True, "loading_verified_fitting_only": True}), flush=True)
    items = load(folders["baseline"], folders["piano"])
    items.extend(mutopia(folders["mutopia"]))
    training, validation = [[i for i in items if i["group"] == group] for group in ("train", "validation")]
    if (len(training) != 781 or len(validation) != 225
            or {i["source_group"] for i in training} & {i["source_group"] for i in validation}
            or sum(i["corpus"] == "musicnet-real-piano" for i in validation) != 15):
        raise ValueError("Changed or leaking fitting partitions")
    arrays, counts = tensors(training)
    coarse, attack, release, x, target, weights = arrays
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
    inputs = [torch.from_numpy(v.astype(np.float32)) for v in (coarse, attack, release, (x-mean)/scale)]
    target, weights = torch.from_numpy(target), torch.from_numpy(weights)
    preserve(output / "supervision.json", {"class_counts_keep_remove": np.bincount(target.numpy(), minlength=2).tolist(),
        "per_recording_class_counts": counts, "training_corpora": dict(Counter(i["corpus"] for i in training)),
        "train_only_normalization": True})
    with np.load(production / "backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        deployed = baseline(validation, ReleaseModel(saved))
    candidates = []
    for profile in PROFILES:
        directory = output / profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(profile["width"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=.00005)
        class_weight = torch.tensor([profile["keep_weight"], 1.], dtype=torch.float32)
        curves, best, memo = [], None, {}
        for epoch in range(1, 41):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(25):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                logits = network(*[v[idx] for v in inputs])
                loss = (torch.nn.functional.cross_entropy(logits, target[idx], weight=class_weight,
                                                          reduction="none")*weights[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite support loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            curve = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic()-started}
            if epoch in STAGES:
                ps = [probabilities(network, item, (mean, scale)) for item in validation]
                checkpoint = directory / f"epoch-{epoch:03d}.pt"
                torch.save({"version": VERSION, "width": profile["width"], "epoch": epoch,
                            "state_dict": network.state_dict(), "mean": torch.from_numpy(mean),
                            "scale": torch.from_numpy(scale)}, checkpoint)
                search, reports = [], []
                for threshold in THRESHOLDS:
                    raw = evaluate(validation, deployed, ps, threshold, memo)
                    strict = evaluate(validation, deployed, ps, min(.9999, threshold+.005), memo)
                    allowed = all(r["passes"] and r["false_notes_removed"] > 0
                                  and r["real_piano_gain"] > 0 for r in (raw, strict))
                    search.append({"eligible": allowed, "raw": {k: v for k, v in raw.items() if k != "per_recording"},
                                   "strict": {k: v for k, v in strict.items() if k != "per_recording"}})
                    reports.append((raw, strict))
                    if allowed:
                        key = (raw["real_piano_gain"], strict["real_piano_gain"], raw["false_notes_removed"],
                               strict["false_notes_removed"], -epoch, threshold)
                        if best is None or key > best[0]:
                            best = (key, {"profile": profile, "epoch": epoch, "threshold": threshold,
                                "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": digest(checkpoint),
                                "plan_sha256": digest(output / "plan.json")}, raw, strict)
                preserve(directory / f"search-{epoch:03d}.json", search)
                diagnostic = max(reports, key=lambda pair: pair[0]["removed_notes"])
                preserve(directory / f"diagnostic-{epoch:03d}.json", {"raw": diagnostic[0], "strict": diagnostic[1]})
                curve["selected"] = best is not None
                curve["safest_setting"] = search[-1]
            curves.append(curve)
            print(json.dumps({"profile": profile["name"], **curve}), flush=True)
        preserve(directory / "learning-curves.json", curves)
        if best is None:
            preserve(directory / "selection.json", {"selected": False, "no_test_used_for_selection": True})
        else:
            _, chosen, raw, strict = best
            preserve(directory / "validation.json", {"raw": raw, "strict": strict})
            chosen["validation_sha256"] = digest(directory / "validation.json")
            preserve(directory / "selection.json", chosen)
            candidates.append(best)
    if (current != deployed_contract(production) or historical != hashes()
            or sources != {name: digest(root / name) for name in SOURCES}):
        raise ValueError("Production or training sources changed during fitting")
    winner = max(candidates, key=lambda value: value[0])[1] if candidates else None
    preserve(output / "batch-selection.json", {"selected": winner is not None, "winner": winner,
        "plan_sha256": digest(output / "plan.json"), "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False})
    print(json.dumps({"complete": True, "winner": winner}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    train(**vars(parser.parse_args()))
