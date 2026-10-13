import copy

import numpy as np
import pytest
from left_ensemble_v61 import select_member, selection_key, union_permissions


def candidate(real=0, sampled=0, pitch=0, *, epoch=25, index=0):
    row = {"veto_threshold": .3}
    for name in ("raw", "strict"):
        row[name] = {"passes": True, "real_piano_gain": real, "nrgcp_gain": sampled,
                     "false_notes_removed": 10, "threshold": .9 if name == "raw" else .905}
        row[name+"_pitch"] = {"passes": True, "wrong_pitch_gain": pitch}
    return {"row": row, "epoch": epoch, "profile_index": index}


def test_each_role_requires_its_own_gain_at_both_policies():
    real, sampled = candidate(real=1, pitch=2), candidate(sampled=1, pitch=3)
    assert select_member([sampled, real], "A") is real
    assert select_member([real, sampled], "B") is sampled
    with pytest.raises(ValueError, match="specialist"):
        select_member([sampled], "A")
    with pytest.raises(ValueError, match="specialist"):
        select_member([real], "B")
    one_policy = candidate(real=10, sampled=10, pitch=10)
    one_policy["row"]["strict"]["nrgcp_gain"] = 0
    assert select_member([one_policy, sampled], "B") is sampled


@pytest.mark.parametrize("name", ["raw", "strict", "raw_pitch", "strict_pitch"])
def test_any_nonpreserving_policy_disqualifies_a_high_gain_member(name):
    bad, good = candidate(100, 100, 100), candidate(1, 1, 1)
    bad["row"][name]["passes"] = False
    assert select_member([bad, good], "A") is good
    assert select_member([bad, good], "B") is good


def test_selection_prioritizes_worst_policy_gain_then_pitch_then_complete_timed_gain():
    high_normal, both = candidate(real=10, pitch=20), candidate(real=2, pitch=1)
    high_normal["row"]["strict"]["real_piano_gain"] = 1
    assert select_member([high_normal, both], "A") is both
    more_pitch = candidate(real=2, pitch=3)
    assert select_member([both, more_pitch], "A") is more_pitch
    more_timed = copy.deepcopy(more_pitch)
    more_timed["row"]["raw"]["false_notes_removed"] = 20
    more_timed["row"]["strict"]["false_notes_removed"] = 20
    assert select_member([more_pitch, more_timed], "A") is more_timed


def test_complete_ties_prefer_earlier_checkpoint_and_original_profile_order():
    early, late = candidate(1, 1, 1), candidate(1, 1, 1, epoch=100)
    later_profile = candidate(1, 1, 1, index=1)
    assert select_member([late, later_profile, early], "A") is early


def test_invalid_role_and_nonfinite_metrics_fail_closed():
    with pytest.raises(ValueError):
        select_member([candidate()], "other")
    bad = candidate(1, 1, 1)
    bad["row"]["strict"]["real_piano_gain"] = float("nan")
    with pytest.raises(ValueError):
        selection_key(bad, "A")


@pytest.mark.parametrize("name,metric", [("strict", "real_piano_gain"), ("strict", "nrgcp_gain"),
    ("strict_pitch", "wrong_pitch_gain"), ("raw", "threshold"), ("strict", "threshold"),
    ("strict", "false_notes_removed")])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_every_policy_metric_is_checked_before_any_minimum_can_hide_it(name, metric, invalid):
    bad = candidate(1, 1, 1)
    bad["row"][name][metric] = invalid
    with pytest.raises(ValueError, match="metrics"):
        selection_key(bad, "A")


def test_union_keeps_both_independent_presence_vetoes():
    rejection = np.array([[.001, .999], [.001, .999]], np.float32)
    a = (rejection, np.array([[0]*40, [1]*40], np.float32))
    b = (rejection, np.array([[1]*40, [1]*40], np.float32))
    policies = [{"threshold": .98, "veto_threshold": .3} for _ in range(2)]
    np.testing.assert_array_equal(union_permissions([a, b], policies)[:, 1], [1, 0])
