"""V44 replacement evaluation against the released V43, with fixed clocks."""

import json

import numpy as np
import torch
from app.services.chord_release import ReleaseModel, correct
from bass_joint_timing import protected
from left_precision_v39 import keep
from left_precision_v43 import gate, model, probabilities
from prepare_robust_training_stems import digest
from score_bass_joint_timing import timing
from score_chord_boundary_v36 import chord_error, excess
from train_melody_timing import matches

VERSION = "original-dynamics-presence-v44-v1"
RELEASE_SHA256 = "ca786763c55aeb8454ba88c0541eb5eedccfea42bdd2429984ebfcc512e7cd54"
ASSET_SHA256 = "38899dd8dbbefafb3a2d5d2f210dc2e6db66087ffc6ee95fcae2635f5d9bbfda"


def deployed_contract(production):
    path = production/"training/results/left-precision-v43-v1/release-checks.json"
    release = json.loads(path.read_text())
    if (digest(path) != RELEASE_SHA256 or release["status"] != "verified"
            or not release["new_weights_deployed"] or not release["no_user_audio_or_scores"]
            or release["future_consumed_regression_recordings"] != 603
            or release["asset_sha256"] != ASSET_SHA256
            or any(digest(production/name) != sha for name, sha in release["source_sha256"].items())
            or any(digest(production/"backend/app/assets"/name) != sha for name, sha in release["arrays"].items())):
        raise ValueError("Production differs from verified V43")
    return {"release_checks_sha256": digest(path), "sources": release["source_sha256"],
            "arrays": release["arrays"], "presence_asset_sha256": ASSET_SHA256}


def released_models(production):
    deployed_contract(production)
    with np.load(production/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        release = ReleaseModel(saved)
    with np.load(production/"backend/app/assets/left-presence-v1.npz", allow_pickle=False) as saved:
        network = model(12)
        template = network.state_dict()
        network.load_state_dict({key: torch.from_numpy(saved["weight_"+key].copy()) for key in template}, strict=True)
        normalizer = tuple(saved[key].copy() for key in ("mean", "scale"))
    network.eval()
    return release, network, normalizer


def baseline(items, models, stricter=False):
    release, network, normalizer = models
    result = []
    for item in items:
        full = correct(item["events"], release.probability(item), item["duration"], release.threshold)
        probability = gate(probabilities(network, item, normalizer), .2*(.8 if stricter else 1))
        accepted = keep(item["events"], probability, item["duration"], .905 if stricter else .9)
        result.append({**item, "events": full[accepted], "full_release_events": full,
                       "baseline_accepted": accepted, "baseline_policy": "strict" if stricter else "raw"})
    return result


def evaluate(items, deployed, ps, threshold, memo=None):
    memo = {} if memo is None else memo
    rows = []
    for original, item, probability in zip(items, deployed, ps, strict=True):
        if original["id"] != item["id"]:
            raise ValueError("Changed V43 baseline identity")
        accepted = keep(original["events"], probability, item["duration"], threshold)
        key = (item["id"], item["baseline_policy"], accepted.tobytes())
        if key not in memo:
            full, before = item["full_release_events"], item["events"]
            old_keep = item["baseline_accepted"]
            if full.shape != original["events"].shape or not np.array_equal(before, full[old_keep]):
                raise ValueError("Changed complete release context")
            after = full[accepted]
            safe, counts, coverage = protected(item, after)
            # Map original V43 identities only; recovered predictions earn no
            # timing credit and deletions retain the same fixed-pair errors.
            common = accepted[old_keep]
            clock = timing(item, before[common], np.flatnonzero(common))
            pairs = matches(item["reference"], before, .2)
            required = {index for _, index in matches(item["reference"], before)}
            required.update(index for _, index in matches(item["reference"], before, offsets=True))
            fixed_identities = all(common[index] for index in required)
            chord_before, chord_pairs = chord_error(item["reference"], before, pairs)
            unsupported_before, unsupported_after = [excess(item["pitch_reference"], e) for e in (before, after)]
            gain = len(before)-counts["attack"]["before"]-len(after)+counts["attack"]["after"]
            memo[key] = {"id": item["id"], "corpus": item["corpus"], "matches": counts,
                "coverage": coverage, "timing": clock, "chord_pairs": chord_pairs,
                "before_chord_spacing_error": chord_before, "after_chord_spacing_error": chord_before,
                "before_unsupported_seconds": unsupported_before, "after_unsupported_seconds": unsupported_after,
                "false_notes_removed": gain, "removed_notes": int(np.sum(old_keep & ~accepted)),
                "recovered_notes": int(np.sum(~old_keep & accepted)),
                "fixed_reference_identities_preserved": bool(fixed_identities),
                "retained_clocks_identical": np.array_equal(before[common], full[old_keep & accepted]),
                "passes": bool(safe and fixed_identities and clock["passes"] and gain >= 0
                    and unsupported_after <= unsupported_before+1e-6)}
        rows.append(memo[key])
    return {"threshold": threshold, "passes": all(row["passes"] for row in rows),
        "failed_recordings": [r["id"] for r in rows if not r["passes"]],
        "false_notes_removed": sum(r["false_notes_removed"] for r in rows),
        "real_piano_gain": sum(r["false_notes_removed"] for r in rows if r["corpus"] == "musicnet-real-piano"),
        "removed_notes": sum(r["removed_notes"] for r in rows),
        "recovered_notes": sum(r["recovered_notes"] for r in rows), "per_recording": rows}
