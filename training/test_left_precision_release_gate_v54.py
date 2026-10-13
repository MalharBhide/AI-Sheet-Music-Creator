import pytest
from left_precision_release_gate_v54 import require_consumed_proof


def proof():
    winner = {"checkpoint_sha256": "checkpoint", "threshold": .98}
    reports = {name: {"passes": True, "threshold": threshold, "false_notes_removed": 603,
        "per_recording": [{"id": f"recording-{i}", "passes": True, "retained_clocks_identical": True,
                           "recovered_notes": 0, "false_notes_removed": 1} for i in range(603)]}
        for name, threshold in (("raw", .98), ("strict", .985))}
    regression = {**reports, "passes": True, "recordings": 603,
        "checkpoint_sha256": "checkpoint", "batch_selection_sha256": "selection",
        "evaluator_sha256": "evaluator", "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    return regression, winner


def test_complete_preserving_evidence_passes():
    regression, winner = proof()
    require_consumed_proof(regression, winner, "selection", "evaluator")


@pytest.mark.parametrize("mutation", ["aggregate", "count", "checkpoint", "selection", "evaluator",
    "incomplete", "duplicate", "different-ids", "timing", "restored", "failed-row", "gain", "threshold", "sum"])
def test_failed_stale_or_incomplete_evidence_rejected(mutation):
    regression, winner = proof()
    row = regression["strict"]["per_recording"][0]
    if mutation == "aggregate":
        regression["passes"] = False
    elif mutation == "count":
        regression["recordings"] = 602
    elif mutation in ("checkpoint", "selection", "evaluator"):
        key = {"checkpoint": "checkpoint_sha256", "selection": "batch_selection_sha256",
               "evaluator": "evaluator_sha256"}[mutation]
        regression[key] = "changed"
    elif mutation == "incomplete":
        regression["strict"]["per_recording"].pop()
    elif mutation == "duplicate":
        row["id"] = "recording-1"
    elif mutation == "different-ids":
        row["id"] = "replacement"
    elif mutation == "timing":
        row["retained_clocks_identical"] = False
    elif mutation == "restored":
        row["recovered_notes"] = 1
    elif mutation == "failed-row":
        row["passes"] = False
    elif mutation == "gain":
        regression["strict"]["false_notes_removed"] = 0
    elif mutation == "threshold":
        regression["strict"]["threshold"] = .98
    elif mutation == "sum":
        row["false_notes_removed"] = 0
    with pytest.raises(ValueError, match="regression"):
        require_consumed_proof(regression, winner, "selection", "evaluator")
