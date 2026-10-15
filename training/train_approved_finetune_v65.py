"""Calibrate the approved final support readout with immutable learned features and release gates."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from current_bass_v32_baseline import hashes
from left_precision_release_gate_v57 import require_validation_proof
from left_precision_v44 import baseline, deployed_contract, released_models
from left_precision_v45 import (
    masked_frame_loss,
)
from left_precision_v46 import evaluate, pitch_evaluate
from left_precision_v59 import (
    absence_candidates,
    frame_supervision,
    gate,
    isolated_negative_weights,
    max_absence_loss,
    probabilities,
    real_piano_weights,
    validate_partitions,
)
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v40 import tensors
from fitting_v63 import FOLDERS, fitting_items, preparation_audit
from train_left_precision_v59 import SOURCES as PREVIOUS_SOURCES
import approved_model_finetune as approved_support
from approved_model_finetune import APPROVED_SHA256, load_approved, storage_preflight, verify_retained_checkpoint
from left_precision_v43 import VERSION as NETWORK_VERSION

VERSION = "approved-final-readout-rehearsal-v65-v1"
FRESH_SEEDS = tuple(range(307101, 307141))

from train_approved_finetune_v64 import SOURCES as APPROVED_SOURCES
from train_approved_finetune_v62 import THRESHOLDS, VETO_THRESHOLDS, STAGES, CHECKPOINT_SHA256
from approved_support_anchor_v64 import teacher_outputs
from approved_head_rehearsal_v65 import freeze_backbone, require_frozen_backbone, keep_rehearsal

PROFILES = (
    {"name": "readout-balanced", "width": 12, "presence_weight": 2, "real_piano_weight": 8,
     "isolated_negative_weight": 16, "max_absence_weight": .25, "learning_rate": .0001,
     "seed": 307001, "freeze_backbone": True, "rehearsal_weight": 4, "hard_keep_weight": 0},
    {"name": "readout-protected", "width": 12, "presence_weight": 4, "real_piano_weight": 8,
     "isolated_negative_weight": 16, "max_absence_weight": .25, "learning_rate": .00003,
     "seed": 307002, "freeze_backbone": True, "rehearsal_weight": 16, "hard_keep_weight": 0},
)
SOURCES = tuple(dict.fromkeys((*APPROVED_SOURCES, "approved_head_rehearsal_v65.py",
    "test_approved_head_rehearsal_v65.py", "train_approved_finetune_v65.py", "test_approved_finetune_v65.py")))


def initialized(production, checkpoint):
    approved = load_approved(production)
    proof = verify_retained_checkpoint(approved, checkpoint, CHECKPOINT_SHA256)
    return approved.network, approved.normalizer, proof


def require_protocol(protocol, sources):
    if (protocol["version"] != VERSION or protocol["profiles"] != list(PROFILES)
            or protocol["source_sha256"] != sources or protocol["approved_asset_sha256"] != APPROVED_SHA256
            or protocol["thresholds"] != list(THRESHOLDS) or protocol["veto_thresholds"] != list(VETO_THRESHOLDS)
            or protocol["fresh_seeds"] != list(FRESH_SEEDS)):
        raise ValueError("Changed approved-model fine-tuning preregistration")


def train(production, output):
    torch.set_num_threads(2)
    current, historical = deployed_contract(production), hashes()
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCES}
    protocol_path = production/"training/results/approved-finetune-v65-v1/preregistration.json"
    protocol = json.loads(protocol_path.read_text())
    require_protocol(protocol, sources)
    if digest(Path(approved_support.__file__)) != sources["approved_model_finetune.py"]:
        raise ValueError("Stale fine-tuning helper in historical reader; restore its frozen version")
    storage = storage_preflight(Path("/training"), 500_000_000,
        audited_links={"left-precision-v43-replay/training/results": production/"training/results"})
    approved_checkpoint = Path("/training/left-precision-v43-v1/calibrated-small/epoch-050.pt")
    teacher, normalizer, initialization = initialized(production, approved_checkpoint)
    audit_sha = preparation_audit()
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "network_version": NETWORK_VERSION, "production": current, "historical_v32": historical,
        "preregistration_sha256": digest(protocol_path), "storage_preflight": storage, "initialization": initialization,
        "source_sha256": sources, "new_preparation_audit_sha256": audit_sha,
        "fitting_manifests": {k: digest(v/"manifest.json") for k, v in FOLDERS.items()},
        "training_recordings": 2302, "validation_recordings": 618,
        "timed_validation_recordings": 381, "pitch_only_validation_recordings": 237,
        "profiles": PROFILES, "epochs": 100, "updates_per_epoch": 40, "batch_size": 256,
        "stages": STAGES, "thresholds": THRESHOLDS, "veto_thresholds": VETO_THRESHOLDS,
        "learning_rate": "per frozen profile; minimum is one tenth of initial", "weight_decay": .003,
        "max_absence_spec": {"interior_start": 8, "interior_stop": 32, "requires_all_known_zero": True,
            "note_class": "REMOVE=1", "pool": "maximum"},
        "later_real_piano_proposals_not_admitted": 49, "hard_keep_fraction": .1, "stricter_margin": .005, "stricter_veto_multiplier": .8,
        "architecture": "Calibrate exactly the existing 13 final support-readout parameters (frame.weight and frame.bias) of the approved 12-channel V43 network. Every other feature extractor, context conditioner and rejection-classifier tensor stays byte-identical. No fresh network, new features, normalization, phase channels or failed checkpoint. Teacher rehearsal protects original KEEP confidence on known frames only; unknown frames and REMOVE candidates are excluded. Existing supervised frame BCE and maximum known-absence loss remain. Deterministic rejection evaluation, original .9/.2 and strict .905/.16 policies, and all release gates remain mandatory.",
        "fitting_loss_heads": ["known-frame-BCE", "known-absence-maximum", "known-KEEP-teacher-rehearsal"],
        "supervision": "All 1221 earlier training recordings plus 712 isolated TinySOL recordings, 144 sampled-piano renders from 48 NRG-CP progression families, 45 PDMX piano MP3 windows, and 108 new full-work piano MP3 derivatives from 36 audited generated chorale parents. Also 72 fixed 24-second piano MP3 windows from five newly audited CC0 generated Keynata parent compositions, entirely fitting-only. These are not human-performance evidence; continuous full-work rendering preserves crop-crossing keys as clipped pitch support without manufactured attacks/releases. Positive Keynata command-support frames are UNKNOWN with zero frame gradient, while known absences remain supervised. The 108 chorale derivatives preserve 4ms authored key commands including holds over four seconds. These commands bound pitch support but do not prove sustained acoustic presence after sample decay: positive frame targets are masked for this corpus and only known absent frames contribute to its frame loss. Candidate KEEP labels conservatively protect key identity; no human hold-duration improvement is claimed. All generated parents and dynamics variants remain fitting-only; no original ensemble recordings or human piano timing claims. The entire new Scarlatti composer group (two CC0 arrangements) is fitting-only. Full-work rendering preserves held keys across fixed 24-second waveform crops: complete key commands supply attack/release references, crossing notes supply clipped pitch support only. No manufactured attacks/releases, no new human timing or instrument-held-out claim; fixed source partitions and earlier instrument/work partitions. All 144 new renders retain whole progression-family partitions; their fixed synthetic instrument is shared. Conservative note KEEP with supplied pitch support. TinySOL provides only isolated fundamental identity: no onset/release or held-duration targets. Fundamental frames inside the source clip are UNKNOWN and have zero loss/gradient; other pitches and authored silence provide absences. Frame loss averages labeled frames only; fully unknown candidates contribute no frame loss. Corpus/recording-balanced note weights. Original approved V43 training-only normalization is preserved exactly and is not refitted on this corpus.",
        "composition": "An additional verifier after V43: intersect complete-input candidate permissions with the corresponding immutable V43 normal/stricter permissions. It is structurally impossible to restore a rejected note. Run unchanged V38 release correction on complete input before applying this intersection. Retained pitch, onset, release and velocity stay bit-identical. Every evaluated recording must have zero recovered predictions.",
        "validation_gate": "Both policies must preserve each of 381 timed recordings' genuine attacks, holds, pitch support, exact matched identities, fixed-pair onset/release timing, repeated spacing, chord timing and unsupported duration. Require positive overall false-note gain and separate positive real-piano gain and positive NRG-CP false-note gain. Separately, all 237 TinySOL instrument-held-out clips must retain every V43 source-fundamental candidate identity and never increase wrong-pitch candidates; positive aggregate wrong-pitch gain required. No invented TinySOL timing metrics. Freeze one validation winner without tests.",
        "subsequent_gate": "All 603 consumed regressions at both policies, then one independent 40-clip original/texture MP3 first pass with seeds 307101-307140 cycling 48/128/192 kbps, separately positive family gain. No runner-up, retuning or replacement clips after failure. Full-context physical/runtime parity and native functional checks before any deployment.",
        "no_user_audio_or_scores": True, "no_audio_inference": True, "new_weights_deployed": False}
    preserve(output/"plan.json", plan)
    print(json.dumps({"plan_frozen": True, "loading_fitting_only": True}), flush=True)
    items = fitting_items()
    training, validation = [[i for i in items if i["group"] == g] for g in ("train", "validation")]
    timed = [i for i in validation if i["corpus"] != "tinysol-isolated-pitch"]
    isolated = [i for i in validation if i["corpus"] == "tinysol-isolated-pitch"]
    validate_partitions(items)
    arrays, counts = tensors(training)
    coarse, attack, release, x, target, weights = arrays
    selected = [i["eligible"] & i["mask"] for i in training]
    supervision = [frame_supervision(i) for i in training]
    frame_target, frame_weight = [torch.from_numpy(np.concatenate([s[h][m] for s, m in zip(supervision, selected, strict=True)])) for h in (0, 1)]
    released = released_models(production)
    mean, scale = normalizer
    inputs = [torch.from_numpy(v.astype(np.float32)) for v in (coarse, attack, release, (x-mean)/scale)]
    if any(len(v) != len(target) for v in (frame_target, frame_weight)):
        raise ValueError("Misaligned masked fitting tensors")
    target, weights = torch.from_numpy(target), torch.from_numpy(weights)
    known_absent = absence_candidates(frame_target, frame_weight, target)
    if not known_absent.any() or torch.any(known_absent & (target == 0)):
        raise ValueError("Missing or invalid known-absence fitting population")
    preserve(output/"supervision.json", {"class_counts_keep_remove": np.bincount(target.numpy(), minlength=2).tolist(),
        "per_recording_class_counts": counts, "training_corpora": dict(Counter(i["corpus"] for i in training)),
        "max_absence_candidates": int(known_absent.sum()),
        "excluded_genuine_keep_candidates": int((target == 0).sum()),
        "excluded_remove_candidates_with_unknown_or_positive_interior": int(((target == 1) & ~known_absent).sum()),
        "labeled_frames": int(frame_weight.sum()), "unknown_frames": int((frame_weight == 0).sum()),
        "normalization": "unchanged approved V43 training-only normalization", "teacher_stability_targets_on_fitting_only": True})
    _, teacher_presence = teacher_outputs(teacher, inputs)
    del teacher
    baselines = {(family, strict): baseline(group, released, stricter=strict)
        for family, group in (("timed", timed), ("pitch", isolated)) for strict in (False, True)}
    candidates = []
    for profile in PROFILES:
        directory = output/profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network, inherited_normalizer, proof = initialized(production, approved_checkpoint)
        if any(not np.array_equal(a,b) for a,b in zip(normalizer,inherited_normalizer,strict=True)):
            raise ValueError("Approved normalization changed between profiles")
        preserve(directory/"initialization.json", proof)
        frozen_backbone = freeze_backbone(network)
        optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],
            lr=profile["learning_rate"], weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=profile["learning_rate"]*.1)
        positive_weight = torch.tensor(profile["presence_weight"], dtype=torch.float32)
        profile_weights = real_piano_weights(training, selected, weights.numpy(), profile["real_piano_weight"])
        profile_weights = torch.from_numpy(isolated_negative_weights(training, selected, profile_weights, profile["isolated_negative_weight"]))
        curves, best, memo = [], None, {}
        for epoch in range(1, 101):
            network.train()
            network.reject.eval()  # Fit the same deterministic rejection function used at runtime.
            losses, started = [], time.monotonic()
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                _, presence = network(*[v[idx] for v in inputs])
                parts = [(masked_frame_loss(presence, frame_target[idx], frame_weight[idx], positive_weight)*profile_weights[idx]).mean(),
                    profile["max_absence_weight"]*max_absence_loss(presence, frame_target[idx], frame_weight[idx], target[idx], profile_weights[idx]),
                    profile["rehearsal_weight"]*keep_rehearsal(presence, teacher_presence[idx], frame_weight[idx], target[idx], profile_weights[idx])]
                loss = sum(parts)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite additional-removal fitting loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
                losses.append([float(p.detach()) for p in parts])
            scheduler.step()
            curve = {"epoch": epoch, "head_losses": np.mean(losses, axis=0).tolist(), "seconds": time.monotonic()-started}
            if epoch in STAGES:
                require_frozen_backbone(network, frozen_backbone)
                heads = {family: [probabilities(network, i, (mean, scale)) for i in group]
                    for family, group in (("timed", timed), ("pitch", isolated))}
                checkpoint = directory/f"epoch-{epoch:03d}.pt"
                torch.save({"version": NETWORK_VERSION, "fine_tune_version": VERSION, "initialized_from_sha256": APPROVED_SHA256, "width": profile["width"], "epoch": epoch, "state_dict": network.state_dict(),
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
                            report[name]["nrgcp_gain"] = sum(r["false_notes_removed"] for r in report[name]["per_recording"]
                                if r["corpus"] == "nrgcp-authored-sampled-piano")
                            report[name+"_pitch"] = pitch_evaluate(isolated, baselines["pitch", strict], ps["pitch", strict], limit)
                        allowed = all(report[n]["passes"] and report[n]["false_notes_removed"] > 0
                            and report[n]["real_piano_gain"] > 0 and report[n]["nrgcp_gain"] > 0
                            and report[n+"_pitch"]["passes"]
                            and report[n+"_pitch"]["wrong_pitch_gain"] > 0 for n in ("raw", "strict"))
                        if allowed:
                            require_validation_proof(report, {"threshold": threshold})
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
        require_frozen_backbone(network, frozen_backbone)
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
    if (current != deployed_contract(production) or historical != hashes() or audit_sha != preparation_audit()
            or sources != {name: digest(root/name) for name in SOURCES}
            or digest(protocol_path) != plan["preregistration_sha256"]
            or any(digest(v/"manifest.json") != plan["fitting_manifests"][k] for k, v in FOLDERS.items())):
        raise ValueError("Production or fitting sources changed during additional-removal fitting")
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
