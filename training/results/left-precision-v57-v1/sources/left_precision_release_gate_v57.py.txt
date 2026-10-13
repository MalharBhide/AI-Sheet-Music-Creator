"""Require complete identity and timing evidence before release-stage access."""

from left_precision_release_gate_v54 import require_consumed_proof as previous_consumed_proof


def require_validation_proof(report, winner):
    cohorts = []
    for name, threshold in (("raw", winner["threshold"]),
                            ("strict", min(.9999, winner["threshold"]+.005))):
        timed, pitch = report[name], report[name+"_pitch"]
        rows, isolated = timed["per_recording"], pitch["per_recording"]
        if (len(rows) != 381 or len({r["id"] for r in rows}) != 381
                or len(isolated) != 237 or len({r["id"] for r in isolated}) != 237
                or timed["threshold"] != threshold or pitch["threshold"] != threshold
                or not timed["passes"] or not pitch["passes"]
                or timed["false_notes_removed"] <= 0 or timed["real_piano_gain"] <= 0
                or timed["nrgcp_gain"] <= 0 or pitch["wrong_pitch_gain"] <= 0
                or pitch["timing_supervision_available"] or pitch["removed_fundamental_candidates"] != 0
                or sum(r["corpus"] == "musicnet-real-piano" for r in rows) != 15
                or sum(r["corpus"] == "nrgcp-authored-sampled-piano" for r in rows) != 36
                or timed["false_notes_removed"] != sum(r["false_notes_removed"] for r in rows)
                or timed["real_piano_gain"] != sum(r["false_notes_removed"] for r in rows if r["corpus"] == "musicnet-real-piano")
                or timed["nrgcp_gain"] != sum(r["false_notes_removed"] for r in rows if r["corpus"] == "nrgcp-authored-sampled-piano")
                or pitch["wrong_pitch_gain"] != sum(r["wrong_pitch_gain"] for r in isolated)
                or any(not r["passes"] or not r["fixed_reference_identities_preserved"]
                       or not r["retained_clocks_identical"] or not r["timing"]["passes"]
                       or not r["coverage"]["passes"] or r["recovered_notes"] != 0 for r in rows)
                or any(not r["passes"] or not r["retained_clocks_identical"] or r["recovered_notes"] != 0
                       or r["removed_fundamental_candidates"] != 0 or r["timing_supervision_available"]
                       or r["corpus"] != "tinysol-isolated-pitch" for r in isolated)):
            raise ValueError("Complete preserving validation evidence required")
        cohorts.append(({r["id"] for r in rows}, {r["id"] for r in isolated}))
    if cohorts[0] != cohorts[1]:
        raise ValueError("Validation policies used different recording cohorts")


def require_consumed_proof(regression, winner, selection_sha, evaluator_sha):
    previous_consumed_proof(regression, winner, selection_sha, evaluator_sha)
    for name in ("raw", "strict"):
        if any(not row["fixed_reference_identities_preserved"] or not row["timing"]["passes"]
               or not row["coverage"]["passes"] for row in regression[name]["per_recording"]):
            raise ValueError("Consumed regression must preserve genuine identity, timing and pitch coverage")
