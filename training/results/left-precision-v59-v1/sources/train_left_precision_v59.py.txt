"""Train fresh phase models with a maximum-frame known-absence objective."""

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
    VERSION,
    absence_candidates,
    frame_supervision,
    gate,
    hard_keep_loss,
    isolated_negative_weights,
    max_absence_loss,
    model,
    probabilities,
    real_piano_weights,
    require_phase_contract,
    validate_partitions,
)
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v40 import tensors
from train_left_precision_v58 import FOLDERS, fitting_items, initial_contract, preparation_audit
from train_left_precision_v58 import SOURCES as PREVIOUS_SOURCES

PROFILES = (
    {"name": "max-absence-balanced", "width": 20, "keep_weight": 24, "presence_weight": 4, "real_piano_weight": 4, "isolated_negative_weight": 8, "hard_keep_weight": 0, "max_absence_weight": 1, "seed": 304001},
    {"name": "max-absence-protected", "width": 20, "keep_weight": 48, "presence_weight": 8, "real_piano_weight": 8, "isolated_negative_weight": 16, "hard_keep_weight": 6, "max_absence_weight": .5, "seed": 304002},
)
THRESHOLDS, VETO_THRESHOLDS, STAGES = (.9, .95, .98, .99, .995, .999), (.05, .1, .2, .3), (25, 50, 100)
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES, "left_precision_v59.py", "train_left_precision_v59.py",
    "test_left_precision_v59.py", "evaluate_left_precision_v59.py", "fresh_left_precision_v59.py")))


def train(production, output):
    torch.set_num_threads(2)
    current, historical = deployed_contract(production), hashes()
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCES}
    audit_sha = preparation_audit()
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "production": current, "historical_v32": historical,
        "source_sha256": sources, "new_preparation_audit_sha256": audit_sha,
        "fitting_manifests": {k: digest(v/"manifest.json") for k, v in FOLDERS.items()},
        "training_recordings": 2302, "validation_recordings": 618,
        "timed_validation_recordings": 381, "pitch_only_validation_recordings": 237,
        "profiles": PROFILES, "epochs": 100, "updates_per_epoch": 40, "batch_size": 256,
        "stages": STAGES, "thresholds": THRESHOLDS, "veto_thresholds": VETO_THRESHOLDS,
        "learning_rate": .0005, "minimum_learning_rate": .00005, "weight_decay": .003,
        "max_absence_spec": {"interior_start": 8, "interior_stop": 32, "requires_all_known_zero": True,
            "note_class": "REMOVE=1", "pool": "maximum"},
        "later_real_piano_proposals_not_admitted": 49, "hard_keep_fraction": .1, "stricter_margin": .005, "stricter_veto_multiplier": .8,
        "architecture": "Fresh 20-channel periodic rejection CNNs and independent boundary-conditioned frame-presence CNNs. The frame head concatenates four fixed candidate-relative channels to the nine physical coarse CQT channels: frame-index rank and three mutually exclusive masks for 8 pre-note, 24 interior and 8 post-note samples. These are sample positions, not uniformly spaced seconds or a tempo grid. Raw 61x18 attack/release encoders and average/max boundary pools still condition the frame head alongside 163 physical context features. Note rejection remains the periodic CNN. Phase buffers are nontrainable and must match the exact contract after serialization. No extra inference features or audio needed; the hard-KEEP term changes fitting only, not inference. The balanced profile has no hard-KEEP auxiliary term; the protected profile averages the worst 10% of supervised KEEP note losses per fitting minibatch on detached loss, weighted 6x. REMOVE candidates never enter the hard-KEEP term. A separate maximum-frame absence penalty applies only to supervised fitting REMOVE candidates with all 24 interior frame targets known zero. Mean softplus of each eligible candidate maximum interior logit, multiplied by the same fitting candidate weights; balanced/protected auxiliary coefficients 1/.5. Genuine KEEP, any UNKNOWN interior and any supplied positive interior fragment are excluded. The maximum objective is aligned with the immutable maximum-interior presence veto; no inference threshold, frame clock or gate changes. Fresh seed initialization for each profile; no inherited or failed member checkpoint, teacher output or failed test fitting data. Training-only real-piano loss weights 4x/8x and isolated TinySOL supervised wrong-pitch candidate weights 8x/16x, each renormalized over fitting candidates. No validation/test labels or corpus identity enter inference. No inherited weights, teacher targets, source identity, absolute pitch, labels or baseline logits as learner inputs. Normalize eligible labeled training candidates only. V43 decisions remain immutable output permissions. Synthetic NRG-CP has progression-held-out but shared-instrument partitions; require separate real-piano gain.",
        "supervision": "All 1221 earlier training recordings plus 712 isolated TinySOL recordings, 144 sampled-piano renders from 48 NRG-CP progression families, 45 PDMX piano MP3 windows, and 108 new full-work piano MP3 derivatives from 36 audited generated chorale parents. Also 72 fixed 24-second piano MP3 windows from five newly audited CC0 generated Keynata parent compositions, entirely fitting-only. These are not human-performance evidence; continuous full-work rendering preserves crop-crossing keys as clipped pitch support without manufactured attacks/releases. Positive Keynata command-support frames are UNKNOWN with zero frame gradient, while known absences remain supervised. The 108 chorale derivatives preserve 4ms authored key commands including holds over four seconds. These commands bound pitch support but do not prove sustained acoustic presence after sample decay: positive frame targets are masked for this corpus and only known absent frames contribute to its frame loss. Candidate KEEP labels conservatively protect key identity; no human hold-duration improvement is claimed. All generated parents and dynamics variants remain fitting-only; no original ensemble recordings or human piano timing claims. The entire new Scarlatti composer group (two CC0 arrangements) is fitting-only. Full-work rendering preserves held keys across fixed 24-second waveform crops: complete key commands supply attack/release references, crossing notes supply clipped pitch support only. No manufactured attacks/releases, no new human timing or instrument-held-out claim; fixed source partitions and earlier instrument/work partitions. All 144 new renders retain whole progression-family partitions; their fixed synthetic instrument is shared. Conservative note KEEP with supplied pitch support. TinySOL provides only isolated fundamental identity: no onset/release or held-duration targets. Fundamental frames inside the source clip are UNKNOWN and have zero loss/gradient; other pitches and authored silence provide absences. Frame loss averages labeled frames only; fully unknown candidates contribute no frame loss. Corpus/recording-balanced note weights. Normalization fitted only on eligible labeled candidates from the 2302 training recordings; never validation.",
        "composition": "An additional verifier after V43: intersect complete-input candidate permissions with the corresponding immutable V43 normal/stricter permissions. It is structurally impossible to restore a rejected note. Run unchanged V38 release correction on complete input before applying this intersection. Retained pitch, onset, release and velocity stay bit-identical. Every evaluated recording must have zero recovered predictions.",
        "validation_gate": "Both policies must preserve each of 381 timed recordings' genuine attacks, holds, pitch support, exact matched identities, fixed-pair onset/release timing, repeated spacing, chord timing and unsupported duration. Require positive overall false-note gain and separate positive real-piano gain and positive NRG-CP false-note gain. Separately, all 237 TinySOL instrument-held-out clips must retain every V43 source-fundamental candidate identity and never increase wrong-pitch candidates; positive aggregate wrong-pitch gain required. No invented TinySOL timing metrics. Freeze one validation winner without tests.",
        "subsequent_gate": "All 603 consumed regressions at both policies, then one independent 40-clip original/texture MP3 first pass with seeds 303101-303140 cycling 48/128/192 kbps, separately positive family gain. No runner-up, retuning or replacement clips after failure. Full-context physical/runtime parity and native functional checks before any deployment.",
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
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .01)
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
        "normalization": "eligible labeled candidates from the 2302 training recordings only", "no_teacher_targets": True})
    baselines = {(family, strict): baseline(group, released, stricter=strict)
        for family, group in (("timed", timed), ("pitch", isolated)) for strict in (False, True)}
    candidates = []
    for profile in PROFILES:
        directory = output/profile["name"]
        directory.mkdir()
        torch.manual_seed(profile["seed"])
        rng = np.random.default_rng(profile["seed"])
        network = model(profile["width"])
        preserve(directory/"initial-permissions.json", initial_contract(network, inputs))
        optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=.00005)
        class_weight = torch.tensor([profile["keep_weight"], 1.], dtype=torch.float32)
        positive_weight = torch.tensor(profile["presence_weight"], dtype=torch.float32)
        profile_weights = real_piano_weights(training, selected, weights.numpy(), profile["real_piano_weight"])
        profile_weights = torch.from_numpy(isolated_negative_weights(training, selected, profile_weights, profile["isolated_negative_weight"]))
        curves, best, memo = [], None, {}
        for epoch in range(1, 101):
            network.train()
            losses, started = [], time.monotonic()
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, len(target), 256))
                rejection, presence = network(*[v[idx] for v in inputs])
                parts = [(torch.nn.functional.cross_entropy(rejection, target[idx], weight=class_weight,
                    reduction="none")*profile_weights[idx]).mean(),
                    (masked_frame_loss(presence, frame_target[idx], frame_weight[idx], positive_weight)*profile_weights[idx]).mean(),
                    profile["hard_keep_weight"]*hard_keep_loss(rejection, target[idx], profile_weights[idx], .1)
                    if profile["hard_keep_weight"] else rejection.sum()*0,
                    profile["max_absence_weight"]*max_absence_loss(presence, frame_target[idx], frame_weight[idx], target[idx], profile_weights[idx])]
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
                heads = {family: [probabilities(network, i, (mean, scale)) for i in group]
                    for family, group in (("timed", timed), ("pitch", isolated))}
                require_phase_contract(network)
                checkpoint = directory/f"epoch-{epoch:03d}.pt"
                torch.save({"version": VERSION, "width": profile["width"], "epoch": epoch, "state_dict": network.state_dict(),
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
        require_phase_contract(network)
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
