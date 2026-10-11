import numpy as np
from chord_boundary_v36 import STAY, context, labels
from protected_boundary_targets_v37 import protected_labels
from score_chord_boundary_v36 import evaluate
from test_chord_boundary_v36 import heads, item


def test_supported_decay_is_not_taught_as_a_premature_release():
    data = item([[3., 4.08, 48, 80]], [[3., 4., 48]])
    data["pitch_reference"] = np.array([[3., 4.2, 48]])
    assert labels(data)[0, 2] == 0
    assert protected_labels(data)[0, 2] == STAY


def test_unsupported_tail_can_still_be_taught_to_shorten():
    data = item([[3., 4.08, 48, 80]], [[3., 4., 48]])
    assert protected_labels(data)[0, 2] == 0


def test_real_longer_hold_can_be_taught_to_extend():
    data = item([[3., 3.92, 48, 80]], [[3., 4., 48]])
    assert protected_labels(data)[0, 2] == 6


def test_protected_targets_do_not_change_observed_model_inputs_or_note_clocks():
    data = item([[3.04, 4., 48, 80], [3., 5., 52, 90]], [[3., 4., 48], [3., 5., 52]])
    before, events = context(data), data["events"].copy()
    target = protected_labels(data)
    assert target[0, 1] != STAY
    np.testing.assert_array_equal(context(data), before)
    np.testing.assert_array_equal(data["events"], events)


def test_timing_target_abstains_when_it_would_worsen_a_repeated_attack_gap():
    data = item([[3.04, 3.8, 48, 80], [4.04, 5., 48, 80]], [[3., 3.8, 48], [4., 5., 48]])
    # The repeated spacing is already exact; moving only one attack breaks it.
    target = protected_labels(data)
    assert target[:, 1].tolist() == [STAY, STAY]


def test_supervised_corrections_individually_pass_the_actual_musical_gate():
    rng = np.random.default_rng(282101)
    for _ in range(12):
        reference = np.array([[3., 4.2, 48], [3., 5.4, 52], [7., 8.5, 48]])
        events = np.column_stack((reference, np.full(3, 80.)))
        events[:, :2] += rng.choice([-.08, -.04, 0., .04, .08], (3, 2))
        data = item(events, reference)
        target = protected_labels(data)
        for index in range(len(events)):
            for head in (1, 2):
                if target[index, head] in (-100, STAY):
                    continue
                ps = heads(len(events))
                ps[head][index] = np.eye(7)[target[index, head]]
                report = evaluate([data], [ps], [np.ones(len(events))], .9, .9, {})
                assert report["passes"], (index, head, report)
