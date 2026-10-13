"""Verify cached synthetic piano labels and physical inputs without new inference."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from left_precision_v40 import observations
from left_precision_v45 import frame_supervision
from prepare_nrgcp_v47 import load
from prepare_robust_training_stems import digest, preserve


def audit(folder, output):
    items = load(folder)
    groups = {g: [i for i in items if i["group"] == g] for g in ("train", "validation")}
    if ({i["source_group"] for i in groups["train"]}
            & {i["source_group"] for i in groups["validation"]}):
        raise ValueError("Leaking progression family")
    rows = []
    for item in items:
        observations(item)  # Validate every input, including zero-prediction clips.
        target, mask = frame_supervision(item)
        if not np.all(mask == 1):
            raise ValueError("Authored piano frames unexpectedly unlabeled")
        reference = item["reference"]
        starts = Counter(reference[:, 0])
        repeats = sum(max(0, int(np.sum(reference[:, 2] == pitch))-1)
                      for pitch in np.unique(reference[:, 2]))
        chosen = item["eligible"] & item["mask"]
        rows.append({"id": item["id"], "group": item["group"], "source_group": item["source_group"],
            "candidate_notes": len(item["events"]), "selected_candidates": int(chosen.sum()),
            "selected_keep": int(np.sum(item["y"][chosen] == 1)),
            "selected_remove": int(np.sum(item["y"][chosen] == 0)),
            "authored_keys": len(reference), "chord_attack_groups": sum(v > 1 for v in starts.values()),
            "same_pitch_recurrences": repeats, "holds_over_one_second": int(np.sum(np.diff(reference[:, :2]) > 1)),
            "holds_over_four_seconds": int(np.sum(np.diff(reference[:, :2]) > 4)),
            "positive_frame_targets": int(target.sum()), "bitrate": item["bitrate"],
            "codec_lag_samples": item["codec_clock"]["codec_lag_samples"],
            "silent_padding_adjustment_samples": item["codec_clock"]["silent_padding_adjustment_samples"]})
    totals = {g: {"recordings": len(subset), "progression_families": len({i["source_group"] for i in subset}),
        "zero_prediction_clips": sum(not len(i["events"]) for i in subset),
        "bitrates": dict(Counter(i["bitrate"] for i in subset)),
        **{k: sum(r[k] for r in rows if r["group"] == g) for k in ("candidate_notes", "selected_candidates",
            "selected_keep", "selected_remove", "authored_keys", "chord_attack_groups", "same_pitch_recurrences",
            "holds_over_one_second", "holds_over_four_seconds")}} for g, subset in groups.items()}
    plan = json.loads((folder/"plan.json").read_text())
    preserve(output, {"source_manifest_sha256": plan["source_manifest_sha256"],
        "preparation_plan_sha256": digest(folder/"plan.json"),
        "preparation_manifest_sha256": digest(folder/"manifest.json"), "auditor_sha256": digest(Path(__file__)),
        "counts": totals, "per_recording": rows, "all_clocks_validated": True,
        "progression_groups_disjoint": True, "shared_synthetic_instrument": True,
        "human_timing_labels": False, "no_new_audio_inference": True,
        "no_reserved_or_consumed_audio_or_metrics": True, "no_user_audio_or_scores": True})
    print(totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    audit(**vars(parser.parse_args()))
