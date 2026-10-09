"""Require real-piano validation gain independently from authored improvements."""

from bass_joint_timing_v18 import margin
from score_declutter_local import evaluate


def real_gain(report, ids):
    rows = [r for r in report["per_recording"] if r["id"] in ids]
    if len(rows) != len(ids) or len({r["id"] for r in rows}) != len(ids):
        raise ValueError("Incomplete real-piano validation identities")
    if any(not r["passes"] for r in rows):
        return -1
    return sum(r["baseline"]["false_positives"] - r["candidate"]["false_positives"] for r in rows)


def choose(items, ps, guardians, thresholds, guardian_thresholds, memo):
    real = {i["id"] for i in items if i["corpus"] == "musicnet-real-piano"}
    if len(real) != 15 or any(i["group"] != "validation" for i in items):
        raise ValueError("Calibrated selection requires exactly 15 real-piano validation excerpts")
    accepted, search = [], []
    for threshold in thresholds:
        for guardian in guardian_thresholds:
            raw = evaluate(items, ps, guardians, threshold, guardian, memo)
            strict = evaluate(items, ps, guardians, margin(threshold), guardian, memo)
            gain, safer = real_gain(raw, real), real_gain(strict, real)
            passes = (
                raw["passes"]
                and strict["passes"]
                and gain > 0
                and safer > 0
                and raw["false_notes_removed"] > 0
                and strict["false_notes_removed"] > 0
            )
            search.append(
                {
                    "threshold": threshold,
                    "guardian_threshold": guardian,
                    "eligible": bool(passes),
                    "raw_passes": raw["passes"],
                    "raw_false_notes_removed": raw["false_notes_removed"],
                    "raw_real_piano_gain": gain,
                    "raw_failed_recordings": raw["failed_recordings"],
                    "margin_passes": strict["passes"],
                    "margin_false_notes_removed": strict["false_notes_removed"],
                    "margin_real_piano_gain": safer,
                    "margin_failed_recordings": strict["failed_recordings"],
                }
            )
            if passes:
                key = (
                    gain,
                    safer,
                    raw["false_notes_removed"],
                    strict["false_notes_removed"],
                    threshold,
                    -guardian,
                )
                accepted.append(
                    (
                        key,
                        {
                            "raw": raw,
                            "margin": strict,
                            "real_piano_gain": gain,
                            "margin_real_piano_gain": safer,
                        },
                    )
                )
    return (max(accepted, key=lambda pair: pair[0])[1] if accepted else None), search
