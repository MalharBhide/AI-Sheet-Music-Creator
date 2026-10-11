"""Require pitch, attack, hold, repeated spacing and chord clock preservation."""

import numpy as np
from bass_joint_timing import protected
from chord_boundary_v36 import decode
from pitch_interval_coverage import reference_coverage
from score_bass_joint_timing import timing
from train_melody_timing import matches


def chord_error(reference, events, pairs):
    error, count = 0., 0
    for slot, (a, i) in enumerate(pairs):
        for b, j in pairs[slot+1:]:
            if reference[a, 2] != reference[b, 2] and abs(reference[a, 0]-reference[b, 0]) <= .025:
                error += abs((events[i, 0]-events[j, 0])-(reference[a, 0]-reference[b, 0]))
                count += 1
    return error, count


def excess(reference, events):
    annotated = np.column_stack((reference, np.ones(len(reference))))
    supported = reference_coverage(events[:, :3], annotated)
    return float(np.maximum(0., events[:, 1]-events[:, 0]-supported).sum())


def evaluate(items, ps, guards, timing_threshold, remove_threshold, memo):
    rows = []
    for item, probability, guardian in zip(items, ps, guards, strict=True):
        old = item["events"]
        after, indices = decode(item, probability, guardian, timing_threshold, remove_threshold)
        key = (item["id"], after.tobytes(), indices.tobytes())
        if key not in memo:
            safe, counts, coverage = protected(item, after)
            clock = timing(item, after, indices)
            paired = matches(item["reference"], old, .2)
            mapped = old.copy()
            mapped[indices] = after  # Deleting a prediction earns no timing credit.
            chord_before, chord_pairs = chord_error(item["reference"], old, paired)
            chord_after, _ = chord_error(item["reference"], mapped, paired)
            before_excess, after_excess = [excess(item["pitch_reference"], e) for e in (old, after)]
            false_before = len(old)-counts["attack"]["before"]
            false_after = len(after)-counts["attack"]["after"]
            memo[key] = {"id": item["id"], "corpus": item["corpus"], "matches": counts,
                         "coverage": coverage, "timing": clock, "chord_pairs": chord_pairs,
                         "before_chord_spacing_error": chord_before, "after_chord_spacing_error": chord_after,
                         "before_unsupported_seconds": before_excess, "after_unsupported_seconds": after_excess,
                         "false_notes_removed": false_before-false_after,
                         "retimed_notes": int(np.any(after[:, :2] != old[indices, :2], axis=1).sum()),
                         "passes": bool(safe and clock["passes"] and false_after <= false_before
                                        and chord_after <= chord_before+1e-9
                                        and after_excess <= before_excess+1e-6)}
        rows.append(memo[key])
    gains = np.sum([np.subtract(r["timing"]["before_onset_release_error_sum"],
                               r["timing"]["after_onset_release_error_sum"]) for r in rows], axis=0)
    return {"passes": all(r["passes"] for r in rows),
            "timing_threshold": timing_threshold, "remove_threshold": remove_threshold,
            "failed_recordings": [r["id"] for r in rows if not r["passes"]],
            "onset_gain_seconds": float(gains[0]), "release_gain_seconds": float(gains[1]),
            "false_notes_removed": sum(r["false_notes_removed"] for r in rows),
            "retimed_notes": sum(r["retimed_notes"] for r in rows), "per_recording": rows}
