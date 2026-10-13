import copy

import pytest
from left_precision_release_gate_v57 import require_consumed_proof, require_validation_proof


def validation():
    reports = {}
    winner = {"threshold": .98}
    for name, threshold in (("raw", .98), ("strict", .985)):
        rows = [{"id": f"timed-{i}", "corpus": "musicnet-real-piano" if i < 15
                 else "nrgcp-authored-sampled-piano" if i < 51 else "earlier",
                 "false_notes_removed": 1, "passes": True, "retained_clocks_identical": True,
                 "fixed_reference_identities_preserved": True, "timing": {"passes": True},
                 "coverage": {"passes": True}, "recovered_notes": 0} for i in range(381)]
        reports[name] = {"per_recording": rows, "threshold": threshold, "passes": True,
                         "false_notes_removed": 381, "real_piano_gain": 15, "nrgcp_gain": 36}
        isolated = [{"id": f"pitch-{i}", "corpus": "tinysol-isolated-pitch", "passes": True,
                     "retained_clocks_identical": True, "recovered_notes": 0,
                     "removed_fundamental_candidates": 0, "timing_supervision_available": False,
                     "wrong_pitch_gain": 1} for i in range(237)]
        reports[name+"_pitch"] = {"per_recording": isolated, "threshold": threshold,
            "passes": True, "wrong_pitch_gain": 237, "removed_fundamental_candidates": 0,
            "timing_supervision_available": False}
    return reports, winner


def test_complete_dual_policy_validation_proof_passes():
    reports, winner = validation()
    require_validation_proof(reports, winner)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different-cohort", "aggregate",
    "real-gain", "nrg-gain", "pitch-gain", "identity", "clock", "timing", "coverage",
    "restored", "threshold", "invented-pitch-timing", "lost-fundamental", "fundamental-sum"])
def test_incomplete_or_contradictory_validation_proof_rejected(mutation):
    reports, winner = validation()
    timed, pitch = reports["strict"], reports["strict_pitch"]
    row = timed["per_recording"][0]
    if mutation == "missing":
        pitch["per_recording"].pop()
    elif mutation == "duplicate":
        row["id"] = timed["per_recording"][1]["id"]
    elif mutation == "different-cohort":
        row["id"] = "replacement"
    elif mutation == "aggregate":
        timed["false_notes_removed"] += 1
    elif mutation in ("real-gain", "nrg-gain"):
        timed[{"real-gain": "real_piano_gain", "nrg-gain": "nrgcp_gain"}[mutation]] += 1
    elif mutation == "pitch-gain":
        pitch["wrong_pitch_gain"] += 1
    elif mutation in ("identity", "clock"):
        row[{"identity": "fixed_reference_identities_preserved", "clock": "retained_clocks_identical"}[mutation]] = False
    elif mutation in ("timing", "coverage"):
        row[mutation]["passes"] = False
    elif mutation == "restored":
        row["recovered_notes"] = 1
    elif mutation == "threshold":
        timed["threshold"] = .98
    elif mutation == "invented-pitch-timing":
        pitch["per_recording"][0]["timing_supervision_available"] = True
    elif mutation == "lost-fundamental":
        pitch["per_recording"][0]["removed_fundamental_candidates"] = 1
    else:
        pitch["removed_fundamental_candidates"] = 1
    with pytest.raises(ValueError, match="validation|Validation"):
        require_validation_proof(reports, winner)


@pytest.mark.parametrize("mutation", [None, "identity", "timing", "coverage"])
def test_consumed_proof_requires_identity_timing_and_coverage_even_if_aggregate_passes(mutation):
    reports, _ = validation()
    template = reports["raw"]["per_recording"][0]
    regression = {"passes": True, "recordings": 603, "checkpoint_sha256": "weights",
                  "batch_selection_sha256": "selection", "evaluator_sha256": "evaluator",
                  "no_test_used_for_selection": True, "no_user_audio_or_scores": True,
                  "new_weights_deployed": False}
    winner = {"threshold": .98, "checkpoint_sha256": "weights"}
    for name, threshold in (("raw", .98), ("strict", .985)):
        rows = [{**copy.deepcopy(template), "id": f"regression-{i}"} for i in range(603)]
        regression[name] = {"passes": True, "threshold": threshold,
                            "false_notes_removed": 603, "per_recording": rows}
    row = regression["strict"]["per_recording"][0]
    if mutation == "identity":
        row["fixed_reference_identities_preserved"] = False
    elif mutation:
        row[mutation]["passes"] = False
    if mutation:
        with pytest.raises(ValueError, match="preserve"):
            require_consumed_proof(regression, winner, "selection", "evaluator")
    else:
        require_consumed_proof(regression, winner, "selection", "evaluator")
