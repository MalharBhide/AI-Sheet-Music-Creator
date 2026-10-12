"""Fine tune released V43 with masked TinySOL pitch labels and fitting rehearsal."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from current_bass_v32_baseline import hashes
from left_precision_v43 import gate, model, probabilities
from left_precision_v44 import baseline, deployed_contract, evaluate, released_models
from left_precision_v45 import (
    VERSION,
    frame_supervision,
    masked_frame_loss,
    pitch_evaluate,
    rehearsal_loss,
)
from musicnet_v32_data import load
from prepare_left_support_v41 import load as authored_items
from prepare_mutopia_recurrence_v42 import load as piano_chord_items
from prepare_nsynth_dynamics_v44 import load as dynamics_items
from prepare_robust_training_stems import digest, preserve
from prepare_tinysol_presence_v45 import load as isolated_items
from train_left_precision_v40 import tensors
from train_left_precision_v44 import FOLDERS as PREVIOUS_FOLDERS
from train_left_precision_v44 import SOURCES as PREVIOUS_SOURCES

PROFILES = (
    {"name": "rehearsal-balanced", "width": 12, "keep_weight": 24, "presence_weight": 4, "anchor_weight": 2, "seed": 290001},
    {"name": "rehearsal-protected", "width": 12, "keep_weight": 48, "presence_weight": 8, "anchor_weight": 4, "seed": 290002},
)
THRESHOLDS, VETO_THRESHOLDS, STAGES = (.9, .95, .98, .99, .995, .999), (.05, .1, .2), (25, 50, 100)
FOLDERS = {**PREVIOUS_FOLDERS, "tinysol": Path("/training/tinysol-presence-v45-observations-v2")}
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES, "acquire_tinysol_v45.py", "prepare_tinysol_presence_v45.py",
    "left_precision_v45.py", "train_left_precision_v45.py", "evaluate_left_precision_v45.py", "fresh_left_precision_v45.py",
    "test_left_precision_v45.py", "test_tinysol_presence.py", "test_tinysol_acquisition.py")))


def fitting_items():
    return (load(FOLDERS["baseline"], FOLDERS["piano"])+authored_items(FOLDERS["authored"])
        +piano_chord_items(FOLDERS["mutopia"])+dynamics_items(FOLDERS["dynamics"])+isolated_items(FOLDERS["tinysol"]))


def train(production, output):
    torch.set_num_threads(2)
    current, historical = deployed_contract(production), hashes()
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCES}
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "production": current, "historical_v32": historical,
        "source_sha256": sources, "fitting_manifests": {k: digest(v/"manifest.json") for k, v in FOLDERS.items()},
        "training_recordings": 1933, "validation_recordings": 582,
        "timed_validation_recordings": 345, "pitch_only_validation_recordings": 237,
        "profiles": PROFILES, "epochs": 100, "updates_per_epoch": 40, "batch_size": 256,
        "stages": STAGES, "thresholds": THRESHOLDS, "veto_thresholds": VETO_THRESHOLDS,
        "learning_rate": .00005, "minimum_learning_rate": .000005, "weight_decay": .003,
        "stricter_margin": .005, "stricter_veto_multiplier": .8,
        "architecture": "Initialize both 12-channel heads and physical normalizer from verified V43; no architecture or inference inputs change. Fine tune on all fitting-only corpora. Rehearse released soft note/frame probabilities only on earlier fitting corpora; teacher is not truth and never supplies validation labels. No source identity, absolute pitch, labels or teacher logits are inference inputs.",
        "supervision": "All 1221 existing training recordings plus 712 isolated TinySOL recordings; fixed source/instrument/work partitions. Conservative note KEEP with supplied pitch support. TinySOL provides only isolated fundamental identity: no onset/release or held-duration targets. Fundamental frames inside the source clip are UNKNOWN and have zero loss/gradient; other pitches and authored silence provide absences. Frame loss averages labeled frames only; fully unknown candidates contribute no frame loss. Corpus/recording-balanced note weights. Frozen V43 normalization was fitted only on past training partitions. No new validation normalization.",
        "composition": "Replacement for V43, evaluated on complete V32 input. Run unchanged V38 release correction with complete context before candidate deletion. Explicitly measure restored wrong notes as regressions. Never alter pitch, attack, release or velocity of retained notes.",
        "validation_gate": "Both policies must preserve each of 345 timed recordings' genuine attacks, holds, pitch support, exact matched identities, fixed-pair onset/release timing, repeated spacing, chord timing and unsupported duration. Require positive overall false-note gain and separate positive real-piano gain. Separately, all 237 TinySOL instrument-held-out clips must retain every V43 source-fundamental candidate identity and never increase wrong-pitch candidates; positive aggregate wrong-pitch gain required. No invented TinySOL timing metrics. Freeze one validation winner without tests.",
        "subsequent_gate": "All 603 consumed regressions at both policies, then one independent 40-clip original/texture MP3 first pass with seeds 290101-290140 cycling 48/128/192 kbps, separately positive family gain. No runner-up, retuning or replacement clips after failure. Full-context physical/runtime parity and native functional checks before any deployment.",
        "no_user_audio_or_scores": True, "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output/"plan.json", plan)
    print(json.dumps({"plan_frozen": True, "loading_fitting_only": True}), flush=True)
    items = fitting_items()
    training, validation = [[i for i in items if i["group"] == g] for g in ("train", "validation")]
    timed = [i for i in validation if i["corpus"] != "tinysol-isolated-pitch"]
    isolated = [i for i in validation if i["corpus"] == "tinysol-isolated-pitch"]
    if (len(training) != 1933 or len(timed) != 345 or len(isolated) != 237
            or len({i["id"] for i in items}) != len(items)
            or {i["source_group"] for i in training} & {i["source_group"] for i in validation}
            or sum(i["corpus"] == "musicnet-real-piano" for i in timed) != 15):
        raise ValueError("Incomplete or leaking weak-pitch fitting partitions")
    arrays, counts = tensors(training)
    coarse, attack, release, x, target, weights = arrays
    selected = [i["eligible"] & i["mask"] for i in training]
    supervision = [frame_supervision(i) for i in training]
    frame_target, frame_weight = [torch.from_numpy(np.concatenate([s[h][m] for s, m in zip(supervision, selected, strict=True)])) for h in (0, 1)]
    released = released_models(production)
    _, teacher, (mean, scale) = released
    inputs = [torch.from_numpy(v.astype(np.float32)) for v in (coarse, attack, release, (x-mean)/scale)]
    teacher_heads = [probabilities(teacher, i, (mean, scale)) for i in training]
    teacher_rejection, teacher_presence = [torch.from_numpy(np.concatenate([p[h][m] for p, m in zip(teacher_heads, selected, strict=True)])) for h in (0, 1)]
    anchor = torch.from_numpy(np.concatenate([np.full(int(m.sum()), i["corpus"] != "tinysol-isolated-pitch", np.float32) for i, m in zip(training, selected, strict=True)]))
    if any(len(v) != len(target) for v in (frame_target, frame_weight, teacher_rejection, teacher_presence, anchor)):
        raise ValueError("Misaligned masked fitting tensors")
    target, weights = torch.from_numpy(target), torch.from_numpy(weights)
    preserve(output/"supervision.json", {"class_counts_keep_remove": np.bincount(target.numpy(), minlength=2).tolist(),
        "per_recording_class_counts": counts, "training_corpora": dict(Counter(i["corpus"] for i in training)),
        "labeled_frames": int(frame_weight.sum()), "unknown_frames": int((frame_weight == 0).sum()),
        "rehearsed_candidates": int(anchor.sum()), "normalization": "unchanged V43 training-only", "no_validation_teacher_labels": True})
    baselines = {(family, strict): baseline(group, released, stricter=strict)
        for family, group in (("timed", timed), ("pitch", isolated)) for strict in (False, True)}
    candidates = []
    for profile in PROFILES:
        directory = output/profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(12)
        network.load_state_dict(teacher.state_dict(), strict=True)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.00005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=.000005)
        class_weight = torch.tensor([profile["keep_weight"], 1.], dtype=torch.float32)
        positive_weight = torch.tensor(profile["presence_weight"], dtype=torch.float32)
        curves, best, memo = [], None, {}
        for epoch in range(1, 101):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                rejection, presence = network(*[v[idx] for v in inputs])
                parts = [(torch.nn.functional.cross_entropy(rejection, target[idx], weight=class_weight,
                    reduction="none")*weights[idx]).mean(),
                    (masked_frame_loss(presence, frame_target[idx], frame_weight[idx], positive_weight)*weights[idx]).mean(),
                    (rehearsal_loss(rejection, presence, teacher_rejection[idx], teacher_presence[idx], frame_weight[idx])
                        *weights[idx]*anchor[idx]*profile["anchor_weight"]).mean()]
                loss = sum(parts)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite masked fine-tuning loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append([float(p.detach()) for p in parts])
            scheduler.step()
            curve = {"epoch": epoch, "head_losses": np.mean(losses, axis=0).tolist(), "seconds": time.monotonic()-started}
            if epoch in STAGES:
                heads = {family: [probabilities(network, i, (mean, scale)) for i in group]
                    for family, group in (("timed", timed), ("pitch", isolated))}
                checkpoint = directory/f"epoch-{epoch:03d}.pt"
                torch.save({"version": VERSION, "width": 12, "epoch": epoch, "state_dict": network.state_dict(),
                    "mean": torch.from_numpy(mean.copy()), "scale": torch.from_numpy(scale.copy())}, checkpoint)
                search, reports = [], []
                for veto in VETO_THRESHOLDS:
                    ps = {(family, strict): [gate(h, veto*(.8 if strict else 1)) for h in values]
                        for family, values in heads.items() for strict in (False, True)}
                    for threshold in THRESHOLDS:
                        report = {}
                        for strict, name in ((False, "raw"), (True, "strict")):
                            limit = min(.9999, threshold+.005) if strict else threshold
                            report[name] = evaluate(timed, baselines["timed", strict], ps["timed", strict], limit, memo)
                            report[name+"_pitch"] = pitch_evaluate(isolated, baselines["pitch", strict], ps["pitch", strict], limit)
                        allowed = all(report[n]["passes"] and report[n]["false_notes_removed"] > 0
                            and report[n]["real_piano_gain"] > 0 and report[n+"_pitch"]["passes"]
                            and report[n+"_pitch"]["wrong_pitch_gain"] > 0 for n in ("raw", "strict"))
                        search.append({"eligible": allowed, "veto_threshold": veto,
                            **{n: {k: v for k, v in r.items() if k != "per_recording"} for n, r in report.items()}})
                        reports.append(report)
                        if allowed:
                            key = tuple(report[n][metric] for metric in ("real_piano_gain", "false_notes_removed") for n in ("raw", "strict"))
                            key += (report["raw_pitch"]["wrong_pitch_gain"], report["strict_pitch"]["wrong_pitch_gain"], -epoch, threshold, -veto)
                            if best is None or key > best[0]:
                                best = (key, {"profile": profile, "epoch": epoch, "threshold": threshold,
                                    "veto_threshold": veto, "checkpoint": str(checkpoint.relative_to(output)),
                                    "checkpoint_sha256": digest(checkpoint), "plan_sha256": digest(output/"plan.json")}, report)
                preserve(directory/f"search-{epoch:03d}.json", search)
                preserve(directory/f"diagnostic-{epoch:03d}.json", max(reports, key=lambda r: r["raw"]["removed_notes"]))
                curve["selected"] = best is not None
                curve["strictest_setting"] = search[len(THRESHOLDS)-1]
            curves.append(curve)
            print(json.dumps({"profile": profile["name"], **curve}), flush=True)
        preserve(directory/"learning-curves.json", curves)
        if best is None:
            preserve(directory/"selection.json", {"selected": False, "no_test_used_for_selection": True})
        else:
            _, chosen, report = best
            preserve(directory/"validation.json", {**report, "veto_threshold": chosen["veto_threshold"],
                "stricter_veto_threshold": chosen["veto_threshold"]*.8})
            chosen["validation_sha256"] = digest(directory/"validation.json")
            preserve(directory/"selection.json", chosen)
            candidates.append(best)
    if (current != deployed_contract(production) or historical != hashes()
            or sources != {name: digest(root/name) for name in SOURCES}
            or any(digest(v/"manifest.json") != plan["fitting_manifests"][k] for k, v in FOLDERS.items())):
        raise ValueError("Production or fitting sources changed during fine tuning")
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
