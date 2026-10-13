import copy

import pytest
from evaluate_consumed_left_ensemble_v61 import regression_qualifies


def report():
    rows = [{"id": f"consumed-{i}", "passes": True, "retained_clocks_identical": True,
             "fixed_reference_identities_preserved": True, "timing": {"passes": True}, "coverage": {"passes": True},
             "false_notes_removed": int(i == 0), "recovered_notes": 0} for i in range(603)]
    return {**{n: {"per_recording": copy.deepcopy(rows), "threshold": .9, "passes": True,
                   "false_notes_removed": 1, "recovered_notes": 0} for n in ("raw", "strict")},
            "recordings": 603, "no_test_used_for_selection": True,
            "no_user_audio_or_scores": True, "new_weights_deployed": False}


def test_complete_consumed_proof_is_required():
    assert regression_qualifies(report())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "different_cohort", "false_gain", "timing", "coverage", "restore", "no_benefit", "user_audio"])
def test_positive_aggregate_cannot_hide_incomplete_or_regressed_consumed_data(kind):
    value = report()
    if kind == "missing":
        value["strict"]["per_recording"].pop()
    elif kind == "duplicate":
        value["raw"]["per_recording"][1]["id"] = "consumed-0"
    elif kind == "different_cohort":
        value["strict"]["per_recording"][0]["id"] = "unrelated"
    elif kind == "false_gain":
        value["raw"]["false_notes_removed"] = 50
    elif kind == "timing":
        value["strict"]["per_recording"][10]["timing"]["passes"] = False
    elif kind == "coverage":
        value["strict"]["per_recording"][10]["coverage"]["passes"] = False
    elif kind == "restore":
        value["raw"]["per_recording"][10]["recovered_notes"] = 1
    elif kind == "no_benefit":
        value["strict"]["false_notes_removed"] = 0
    else:
        value["no_user_audio_or_scores"] = False
    assert not regression_qualifies(value)
