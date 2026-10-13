"""Train fresh dynamics-aware replacements; compare to live V43, not V38."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from current_bass_v32_baseline import hashes
from left_precision_v43 import gate, model, presence_targets, probabilities
from left_precision_v44 import VERSION, baseline, deployed_contract, evaluate, released_models
from musicnet_v32_data import load
from prepare_left_support_v41 import load as authored_items
from prepare_mutopia_recurrence_v42 import load as piano_chord_items
from prepare_nsynth_dynamics_v44 import load as dynamics_items
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v40 import tensors
from train_left_precision_v43 import SOURCES as PRIOR_SOURCES

PROFILES = (
    {"name": "dynamics-small", "width": 12, "keep_weight": 16, "presence_weight": 2, "seed": 289001},
    {"name": "dynamics-protected", "width": 20, "keep_weight": 24, "presence_weight": 4, "seed": 289002},
)
THRESHOLDS = (.9, .95, .98, .99, .995, .999)
STAGES = (25, 50, 100)
VETO_THRESHOLDS = (.05, .1, .2, .3)
FOLDERS = {"baseline": Path("/training/bass-v32-baseline-v1"),
    "piano": Path("/training/musicnet-v32-observations-v1"),
    "authored": Path("/training/left-support-v41-observations"),
    "mutopia": Path("/training/mutopia-recurrence-v42-observations"),
    "dynamics": Path("/training/nsynth-dynamics-v44-observations")}
SOURCES = tuple(dict.fromkeys((*PRIOR_SOURCES, "acquire_nsynth_dynamics_v44.py", "prepare_nsynth_dynamics_v44.py",
    "left_precision_v44.py", "train_left_precision_v44.py", "evaluate_left_precision_v44.py", "fresh_left_precision_v44.py")))


def train(production, output):
    torch.set_num_threads(2)
    current, historical = deployed_contract(production), hashes()
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCES}
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "production": current, "historical_v32": historical,
        "source_sha256": sources,
        "fitting_manifests": {k: digest(v/"manifest.json") for k, v in FOLDERS.items()},
        "training_recordings": 1221, "validation_recordings": 345, "profiles": PROFILES,
        "epochs": 100, "updates_per_epoch": 40, "batch_size": 256, "stages": STAGES,
        "learning_rate": .0005, "minimum_learning_rate": .00005, "weight_decay": .003,
        "thresholds": THRESHOLDS, "stricter_margin": .005,
        "veto_thresholds": VETO_THRESHOLDS, "stricter_veto_multiplier": .8,
        "architecture": "Fresh spectral/periodic rejection CNN and separate convolutional 40-frame pitch-presence head conditioned on 163 physical context features. Note-level rejection and per-frame presence are calibrated separately on validation: removal needs high rejection confidence and every central frame below a predeclared veto threshold. The stricter policy raises rejection by .005 and lowers the presence veto by 20 percent. No inherited weights, fitted logits, reference labels, absolute pitch, filename or source identity as inference inputs.",
        "supervision": "Conservative note KEEP labels plus independent frame presence from supplied pitch-support intervals using the exact coarse-window clock. Half-hop boundary padding; promote any sub-frame supplied overlap to its nearest central sample, and preserve jittered attack/hold labels. Train-only normalization and corpus/recording weights. Weighted note cross-entropy and frame binary cross-entropy. All existing 88/24 authored NSynth and 16/4 public-domain Mutopia fitting clips, plus 352/96 new original-velocity NSynth MP3 clips covering all eight articulations at velocities 25/50/100/127. Original instrument/work partitions unchanged. No reserved clips fitted or labels guessed from audio.",
        "composition": "Fresh replacement for the V43 presence verifier. Compute candidate decisions from complete V32 input, run unchanged V38 release correction on complete input, then apply candidate deletions. Compare normal/stricter candidates to the corresponding frozen V43 normal/stricter policy. Recoveries of previously deleted notes are evaluated explicitly, including false-note and unsupported-duration regressions. No onset, release, pitch or velocity edits.",
        "validation_gate": "Every recording must preserve attacks, holds, pitch support, fixed-pair onset/release timing, repeated spacing, chord timing and unsupported duration. Baseline V43 predictions matched to reference attacks or holds must retain their exact identities, without substituting a nearby prediction. Positive overall false-note reduction and separately positive MusicNet real-piano gain at normal and stricter thresholds. Freeze one validation winner without any test use.",
        "subsequent_gate": "All 603 consumed regressions at both thresholds, followed by one independent 40-clip original/texture MP3 first pass with seeds 289101-289140 cycling 48/128/192 kbps and separately positive gain. Never fit consumed failures, retune, choose a runner-up or replace a first-pass clip after failure. Export, complete physical/runtime parity and native functional tests before website changes.",
        "no_user_audio_or_scores": True, "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output/"plan.json", plan)
    print(json.dumps({"plan_frozen": True, "loading_fitting_only": True}), flush=True)
    items = load(FOLDERS["baseline"], FOLDERS["piano"])+authored_items(FOLDERS["authored"])+piano_chord_items(FOLDERS["mutopia"])+dynamics_items(FOLDERS["dynamics"])
    training, validation = [[i for i in items if i["group"] == g] for g in ("train", "validation")]
    if (len(training) != 1221 or len(validation) != 345 or len({i["id"] for i in items}) != len(items)
            or {i["source_group"] for i in training} & {i["source_group"] for i in validation}
            or sum(i["corpus"] == "musicnet-real-piano" for i in validation) != 15):
        raise ValueError("Incomplete or leaking dual-head fitting partitions")
    arrays, counts = tensors(training)
    coarse, attack, release, x, target, weights = arrays
    frame_target = torch.from_numpy(np.concatenate([presence_targets(i)[i["eligible"] & i["mask"]] for i in training]))
    if len(frame_target) != len(target):
        raise ValueError("Misaligned frame targets")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
    inputs = [torch.from_numpy(v.astype(np.float32)) for v in (coarse, attack, release, (x-mean)/scale)]
    target, weights = torch.from_numpy(target), torch.from_numpy(weights)
    preserve(output/"supervision.json", {"class_counts_keep_remove": np.bincount(target.numpy(), minlength=2).tolist(),
        "per_recording_class_counts": counts, "training_corpora": dict(Counter(i["corpus"] for i in training)),
        "train_only_normalization": True})
    released = released_models(production)
    deployed = baseline(validation, released)
    stricter_deployed = baseline(validation, released, stricter=True)
    candidates = []
    for profile in PROFILES:
        directory = output/profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(profile["width"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=.00005)
        class_weight = torch.tensor([profile["keep_weight"], 1.], dtype=torch.float32)
        presence_weight = torch.tensor(profile["presence_weight"], dtype=torch.float32)
        curves, best, memo = [], None, {}
        for epoch in range(1, 101):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                rejection, presence = network(*[v[idx] for v in inputs])
                parts = [(torch.nn.functional.cross_entropy(rejection, target[idx], weight=class_weight,
                          reduction="none")*weights[idx]).mean(),
                         (torch.nn.functional.binary_cross_entropy_with_logits(presence, frame_target[idx],
                          pos_weight=presence_weight, reduction="none")*weights[idx, None]).mean()]
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
                heads = [probabilities(network, item, (mean, scale)) for item in validation]
                checkpoint = directory/f"epoch-{epoch:03d}.pt"
                torch.save({"version": VERSION, "width": profile["width"], "epoch": epoch,
                    "state_dict": network.state_dict(), "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)}, checkpoint)
                search, reports = [], []
                for veto in VETO_THRESHOLDS:
                    ps = [gate(h, veto) for h in heads]
                    stricter_ps = [gate(h, veto*.8) for h in heads]
                    for threshold in THRESHOLDS:
                        raw = evaluate(validation, deployed, ps, threshold, memo)
                        strict = evaluate(validation, stricter_deployed, stricter_ps, min(.9999, threshold+.005), memo)
                        allowed = all(r["passes"] and r["false_notes_removed"] > 0 and r["real_piano_gain"] > 0 for r in (raw, strict))
                        search.append({"eligible": allowed, "veto_threshold": veto, "raw": {k: v for k, v in raw.items() if k != "per_recording"},
                            "strict": {k: v for k, v in strict.items() if k != "per_recording"}})
                        reports.append((raw, strict))
                        if allowed:
                            key = (raw["real_piano_gain"], strict["real_piano_gain"], raw["false_notes_removed"],
                                   strict["false_notes_removed"], -epoch, threshold, -veto)
                            if best is None or key > best[0]:
                                best = (key, {"profile": profile, "epoch": epoch, "threshold": threshold, "veto_threshold": veto,
                                    "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": digest(checkpoint),
                                    "plan_sha256": digest(output/"plan.json")}, raw, strict)
                preserve(directory/f"search-{epoch:03d}.json", search)
                diagnostic = max(reports, key=lambda pair: pair[0]["removed_notes"])
                preserve(directory/f"diagnostic-{epoch:03d}.json", {"raw": diagnostic[0], "strict": diagnostic[1]})
                curve["selected"] = best is not None
                curve["strictest_setting"] = search[len(THRESHOLDS)-1]
            curves.append(curve)
            print(json.dumps({"profile": profile["name"], **curve}), flush=True)
        preserve(directory/"learning-curves.json", curves)
        if best is None:
            preserve(directory/"selection.json", {"selected": False, "no_test_used_for_selection": True})
        else:
            _, chosen, raw, strict = best
            preserve(directory/"validation.json", {"raw": raw, "strict": strict,
                "veto_threshold": chosen["veto_threshold"], "stricter_veto_threshold": chosen["veto_threshold"]*.8})
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
