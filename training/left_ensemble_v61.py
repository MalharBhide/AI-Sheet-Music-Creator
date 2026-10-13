"""One fixed validation-selected pair of fresh phase-model specialists."""

import numpy as np
from left_ensemble_v56 import union_permissions

VERSION = "real-and-sampled-piano-phase-union-v61-v1"
FRESH_SEEDS = tuple(range(304101, 304141))
OUTPUT_THRESHOLD = .9
__all__ = ["VERSION", "FRESH_SEEDS", "OUTPUT_THRESHOLD", "union_permissions", "selection_key", "select_member"]


def selection_key(candidate, role):
    if role not in ("A", "B"):
        raise ValueError("Unknown preregistered phase-member role")
    row = candidate["row"]
    observed = [row[name][metric] for name in ("raw", "strict")
                for metric in ("real_piano_gain", "nrgcp_gain", "false_notes_removed", "threshold")]
    observed.extend(row[name+"_pitch"]["wrong_pitch_gain"] for name in ("raw", "strict"))
    observed.extend((row["veto_threshold"], candidate["epoch"], candidate["profile_index"]))
    # Validate before reducing: Python min(1, NaN) may return 1 and hide the NaN.
    if not np.isfinite(observed).all():
        raise ValueError("Invalid phase-member validation metrics")
    primary = "real_piano_gain" if role == "A" else "nrgcp_gain"
    key = (min(row[name][primary] for name in ("raw", "strict")),
           min(row[name+"_pitch"]["wrong_pitch_gain"] for name in ("raw", "strict")),
           min(row[name]["false_notes_removed"] for name in ("raw", "strict")),
           sum(row[name]["false_notes_removed"] for name in ("raw", "strict")),
           row["raw"]["threshold"], -row["veto_threshold"], -candidate["epoch"], -candidate["profile_index"])
    if not np.isfinite(key).all():
        raise ValueError("Invalid phase-member validation metrics")
    return key


def select_member(candidates, role):
    if role not in ("A", "B"):
        raise ValueError("Unknown preregistered phase-member role")
    primary = "real_piano_gain" if role == "A" else "nrgcp_gain"
    preserving = [candidate for candidate in candidates
                  if all(candidate["row"][name]["passes"] is True
                         for name in ("raw", "strict", "raw_pitch", "strict_pitch"))
                  and all(candidate["row"][name][primary] > 0 for name in ("raw", "strict"))]
    if not preserving:
        raise ValueError("No preserving fixed-source specialist; no replacement source")
    return max(preserving, key=lambda candidate: selection_key(candidate, role))
