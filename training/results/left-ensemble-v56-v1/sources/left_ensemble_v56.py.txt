"""Preregistered union of two trained, independently vetoed removal permissions."""

import numpy as np
from left_precision_v43 import gate

VERSION = "new-piano-and-wrong-pitch-specialist-union-v56-v1"
FRESH_SEEDS = tuple(range(300101, 300141))
OUTPUT_THRESHOLD = .9  # Existing evaluator accepts .8–1; binary outputs are invariant.


def union_permissions(heads, policies, stricter=False):
    if len(heads) != 2 or len(policies) != 2 or not isinstance(stricter, bool):
        raise ValueError("Exactly two fixed member policies required")
    decisions = []
    for values, policy in zip(heads, policies, strict=True):
        threshold, veto = policy["threshold"], policy["veto_threshold"]
        if not np.isfinite(threshold) or not .8 <= threshold <= 1:
            raise ValueError("Invalid member rejection threshold")
        probability = gate(values, veto*(.8 if stricter else 1))
        limit = min(.9999, threshold+.005) if stricter else threshold
        decisions.append(probability[:, 1] >= limit)
    if decisions[0].shape != decisions[1].shape:
        raise ValueError("Misaligned complete member candidates")
    remove = decisions[0] | decisions[1]
    return np.column_stack((~remove, remove)).astype(np.float32)


def selection_key(candidate, role):
    if role not in ("A", "B"):
        raise ValueError("Unknown preregistered member role")
    row = candidate["row"]
    minima = {metric: min(row[n][metric] for n in ("raw", "strict"))
              for metric in ("real_piano_gain", "nrgcp_gain", "false_notes_removed")}
    pitch = min(row[n+"_pitch"]["wrong_pitch_gain"] for n in ("raw", "strict"))
    first = (minima["real_piano_gain"], minima["nrgcp_gain"]) if role == "A" else (pitch, minima["false_notes_removed"])
    key = (*first, minima["nrgcp_gain"], minima["false_notes_removed"],
           sum(row[n]["false_notes_removed"] for n in ("raw", "strict")),
           row["raw"]["threshold"], -row["veto_threshold"],
           -candidate["epoch"], -candidate["profile_index"])
    if not np.isfinite(key).all():
        raise ValueError("Invalid member selection metrics")
    return key


def select_member(candidates, role):
    if role not in ("A", "B"):
        raise ValueError("Unknown preregistered member role")
    preserving = [c for c in candidates if all(c["row"][n]["passes"] is True
                  for n in ("raw", "strict", "raw_pitch", "strict_pitch"))]
    preserving = [c for c in preserving if all(
        (c["row"][n]["real_piano_gain"] > 0 and c["row"][n]["nrgcp_gain"] > 0)
        if role == "A" else c["row"][n+"_pitch"]["wrong_pitch_gain"] > 0
        for n in ("raw", "strict"))]
    if not preserving:
        raise ValueError("No preserving member; no replacement source allowed")
    return max(preserving, key=lambda c: selection_key(c, role))
