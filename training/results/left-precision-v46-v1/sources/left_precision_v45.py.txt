"""Masked pitch-only fine tuning and validation without invented note clocks."""

import numpy as np
from left_precision_v39 import keep
from left_precision_v43 import presence_targets

VERSION = "licensed-weak-pitch-finetune-v45-v1"


def frame_supervision(item):
    if item.get("timing_supervision_available") is False:
        if item["corpus"] != "tinysol-isolated-pitch":
            raise ValueError("Unknown weak-pitch corpus")
        target, weight = (np.asarray(item[k], np.float32) for k in ("frame_target", "frame_weight"))
    else:
        target = presence_targets(item)
        weight = np.ones_like(target)
    if (target.shape != (len(item["events"]), 40) or weight.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(weight).all()
            or np.any((target < 0) | (target > 1)) or np.any((weight != 0) & (weight != 1))):
        raise ValueError("Invalid masked pitch supervision")
    return target, weight


def masked_frame_loss(logits, targets, mask, positive_weight):
    """Each candidate averages only labeled frames; all-unknown has zero loss."""
    import torch

    if logits.shape != targets.shape or mask.shape != logits.shape or logits.ndim != 2:
        raise ValueError("Misaligned frame supervision")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight, reduction="none")
    return (loss*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)


def rehearsal_loss(rejection, presence, teacher_rejection, teacher_presence, mask):
    """Teacher targets are fitting predictions only; unknown frames stay masked."""
    import torch

    note = torch.nn.functional.kl_div(torch.log_softmax(rejection, dim=1),
        teacher_rejection, reduction="none").sum(dim=1)
    frame = masked_frame_loss(presence, teacher_presence, mask, None)
    # Subtract teacher entropy: the loss is zero at the released prediction.
    safe = teacher_presence.clamp(1e-7, 1-1e-7)
    entropy = -(teacher_presence*safe.log()+(1-teacher_presence)*(1-safe).log())
    frame -= (entropy*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
    return note+frame


def pitch_evaluate(items, deployed, probabilities, threshold):
    """Protect exact baseline fundamental identities; no attack/hold score."""
    rows = []
    for original, item, probability in zip(items, deployed, probabilities, strict=True):
        if (original["id"] != item["id"] or original["corpus"] != "tinysol-isolated-pitch"
                or original.get("timing_supervision_available") is not False):
            raise ValueError("Invalid weak-pitch evaluation identity")
        accepted = keep(original["events"], probability, original["duration"], threshold)
        previous, full = item["baseline_accepted"], item["full_release_events"]
        supplied = original["y"].astype(bool)
        if (previous.shape != supplied.shape or full.shape != original["events"].shape
                or not np.array_equal(item["events"], full[previous])):
            raise ValueError("Changed weak-pitch baseline context")
        removed_true = int(np.sum(previous & supplied & ~accepted))
        before = int(np.sum(previous & ~supplied))
        after = int(np.sum(accepted & ~supplied))
        common = previous & accepted
        identical = np.array_equal(item["events"][accepted[previous]], full[common])
        rows.append({"id": original["id"], "corpus": original["corpus"],
            "source_group": original["source_group"], "baseline_fundamental_candidates": int(np.sum(previous & supplied)),
            "removed_fundamental_candidates": removed_true, "before_wrong_pitch_candidates": before,
            "after_wrong_pitch_candidates": after, "wrong_pitch_gain": before-after,
            "recovered_notes": int(np.sum(~previous & accepted)),
            "retained_clocks_identical": bool(identical), "timing_supervision_available": False,
            "passes": removed_true == 0 and after <= before and bool(identical)})
    return {"threshold": threshold, "passes": all(r["passes"] for r in rows),
        "failed_recordings": [r["id"] for r in rows if not r["passes"]],
        "wrong_pitch_gain": sum(r["wrong_pitch_gain"] for r in rows),
        "removed_fundamental_candidates": sum(r["removed_fundamental_candidates"] for r in rows),
        "timing_supervision_available": False, "per_recording": rows}
