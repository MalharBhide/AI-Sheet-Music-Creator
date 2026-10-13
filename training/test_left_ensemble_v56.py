import numpy as np
import pytest
from left_ensemble_v56 import OUTPUT_THRESHOLD, select_member, union_permissions
from left_precision_v39 import keep
from left_precision_v46 import permissions


def heads(reject, presence):
    reject = np.asarray(reject, np.float32)
    return np.column_stack((1-reject, reject)), np.full((len(reject), 40), presence, np.float32)


def test_union_respects_each_independent_frame_veto_and_immutable_baseline():
    policy = {"threshold": .9, "veto_threshold": .3}
    a, b = heads([.99, .99, .1], 0), heads([.1, .99, .99], 1)
    result = union_permissions([a, b], [policy, policy])
    np.testing.assert_array_equal(result[:, 1], [1, 1, 0])
    baseline = np.array([True, False, False])
    locked = permissions(result, baseline)
    np.testing.assert_array_equal(locked[:, 1], [1, 1, 1])


def test_stricter_member_policy_applies_both_threshold_and_presence_margin():
    policy = {"threshold": .9, "veto_threshold": .3}
    a, b = heads([.902, .99], .25), heads([.1, .1], 1)
    np.testing.assert_array_equal(union_permissions([a, b], [policy, policy])[:, 1], [1, 1])
    np.testing.assert_array_equal(union_permissions([a, b], [policy, policy], True)[:, 1], [0, 0])


def test_binary_output_decisions_are_invariant_under_supported_evaluator_thresholds():
    events = np.array([[3., 4., 40., 80.], [5., 6., 40., 80.], [7., 8., 80., 80.]])
    policy = {"threshold": .9, "veto_threshold": .3}
    values = heads([.99, .1, .99], 0)
    result = union_permissions([values, values], [policy, policy])
    expected = keep(events, result, 30., OUTPUT_THRESHOLD)
    for threshold in (.8, .9, .995, 1.):
        np.testing.assert_array_equal(keep(events, result, 30., threshold), expected)
    np.testing.assert_array_equal(expected, [False, True, True])


@pytest.mark.parametrize("kind", ["missing_member", "misaligned", "bad_threshold", "nonfinite"])
def test_invalid_member_inputs_fail_closed(kind):
    a, b = heads([.99], 0), heads([.99], 0)
    policies = [{"threshold": .9, "veto_threshold": .3} for _ in range(2)]
    values = [a, b]
    if kind == "missing_member":
        values.pop()
    elif kind == "misaligned":
        values[1] = heads([.99, .99], 0)
    elif kind == "bad_threshold":
        policies[0]["threshold"] = .5
    else:
        values[0][0][0, 0] = np.nan
    with pytest.raises(ValueError):
        union_permissions(values, policies)


def candidate(pitch, real, *, passes=True, epoch=25):
    row = {"veto_threshold": .3}
    for n in ("raw", "strict"):
        row[n] = {"passes": passes, "real_piano_gain": real, "nrgcp_gain": 1,
                  "false_notes_removed": 10, "threshold": .9}
        row[n+"_pitch"] = {"passes": passes, "wrong_pitch_gain": pitch}
    return {"row": row, "epoch": epoch, "profile_index": 0}


def test_roles_use_frozen_priorities_and_never_select_nonpreserving_member():
    pitch, real, invalid = candidate(1, 0), candidate(0, 2), candidate(100, 100, passes=False)
    assert select_member([pitch, real, invalid], "A") is real
    assert select_member([pitch, real, invalid], "B") is pitch
    with pytest.raises(ValueError):
        select_member([invalid], "A")
    with pytest.raises(ValueError):
        select_member([pitch], "C")
    with pytest.raises(ValueError):
        select_member([pitch], "A")
    with pytest.raises(ValueError):
        select_member([real], "B")


def test_equal_metrics_prefer_earlier_checkpoint():
    early, late = candidate(1, 1), candidate(1, 1, epoch=100)
    assert select_member([late, early], "A") is early
