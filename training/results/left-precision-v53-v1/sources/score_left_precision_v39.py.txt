"""Evaluate removal against deployed V38 clocks, without retiming retained notes."""

import numpy as np
from app.services.chord_release import correct
from bass_joint_timing import protected
from left_precision_v39 import keep
from score_bass_joint_timing import timing
from score_chord_boundary_v36 import chord_error, excess
from train_melody_timing import matches


def baseline(items, release):
    return [{**item, "events": correct(item["events"], release.probability(item),
                                       item["duration"], release.threshold)} for item in items]


def evaluate(items, deployed, ps, threshold, memo=None):
    memo = {} if memo is None else memo
    rows = []
    for original, item, probability in zip(items, deployed, ps, strict=True):
        if original["id"] != item["id"]:
            raise ValueError("Changed deployed observation identity")
        accepted = keep(original["events"], probability, item["duration"], threshold)
        key = (item["id"], accepted.tobytes())
        if key not in memo:
            old, after = item["events"], item["events"][accepted]
            safe, counts, coverage = protected(item, after)
            clock = timing(item, after, np.flatnonzero(accepted))
            pairs = matches(item["reference"], old, .2)
            chord_before, chord_pairs = chord_error(item["reference"], old, pairs)
            # Mapped clocks are bit-identical: deletions cannot earn timing credit.
            chord_after, _ = chord_error(item["reference"], old, pairs)
            unsupported_before, unsupported_after = [excess(item["pitch_reference"], e) for e in (old, after)]
            gain = len(old)-counts["attack"]["before"]-len(after)+counts["attack"]["after"]
            memo[key] = {"id": item["id"], "corpus": item["corpus"], "matches": counts,
                "coverage": coverage, "timing": clock, "chord_pairs": chord_pairs,
                "before_chord_spacing_error": chord_before, "after_chord_spacing_error": chord_after,
                "before_unsupported_seconds": unsupported_before, "after_unsupported_seconds": unsupported_after,
                "false_notes_removed": gain, "removed_notes": int((~accepted).sum()),
                "retained_clocks_identical": np.array_equal(after, old[np.flatnonzero(accepted)]),
                "passes": bool(safe and clock["passes"] and gain >= 0
                    and unsupported_after <= unsupported_before+1e-6)}
        rows.append(memo[key])
    return {"threshold": threshold, "passes": all(r["passes"] for r in rows),
            "failed_recordings": [r["id"] for r in rows if not r["passes"]],
            "false_notes_removed": sum(r["false_notes_removed"] for r in rows),
            "real_piano_gain": sum(r["false_notes_removed"] for r in rows if r["corpus"] == "musicnet-real-piano"),
            "removed_notes": sum(r["removed_notes"] for r in rows),
            "per_recording": rows}
