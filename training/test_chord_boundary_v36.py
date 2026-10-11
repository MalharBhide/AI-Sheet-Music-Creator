import numpy as np
import pytest
from chord_boundary_v36 import (
    SHIFTS,
    STAY,
    chord_context,
    decode,
    labels,
    model,
    probabilities,
    transform,
)
from score_chord_boundary_v36 import evaluate


def item(events, reference=None):
    events = np.asarray(events, float).reshape(-1, 4)
    reference = events[:, :3].copy() if reference is None else np.asarray(reference, float).reshape(-1, 3)
    return {"id": "fixture", "corpus": "original", "events": events, "reference": reference,
            "pitch_reference": reference.copy(), "eligible": np.ones(len(events), bool),
            "duration": 20., "mask": np.ones(len(events), bool), "y": np.ones(len(events)),
            "x": np.zeros((len(events), 84), np.float32), "release_neighbors": np.zeros((len(events), 8), np.float32),
            "attack_frames": np.zeros((len(events), 61, 18), np.float32),
            "release_frames": np.zeros((len(events), 61, 18), np.float32)}


def heads(n):
    output = tuple(np.zeros((n, size)) for size in (2, 7, 7))
    for head, stay in zip(output, (0, STAY, STAY), strict=True):
        head[:, stay] = 1
    return output


def test_observed_chord_context_is_order_independent_and_has_no_label_inputs():
    events = np.array([[3., 5., 48, 80], [3.01, 6., 52, 90], [6., 7., 48, 80]])
    order = np.array([2, 0, 1])
    np.testing.assert_array_equal(chord_context(events)[order], chord_context(events[order]))
    assert chord_context(events)[0, 0] == .125
    assert chord_context(events)[0, 1] == 0  # These chord tones release separately.
    assert chord_context(np.empty((0, 4))).shape == (0, 8)


def test_boundaries_have_independent_reference_targets_and_abstain_outside_window():
    data = item([[3., 4., 48, 80], [5., 6., 52, 90]], [[2.96, 4.08, 48], [5., 7., 52]])
    y = labels(data)
    assert y[0].tolist() == [0, 1, 6]
    assert y[1].tolist() == [0, STAY, -100]
    data["eligible"][:] = False
    assert np.all(labels(data) == -100)


def test_chord_tones_are_never_forced_to_share_a_release():
    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    after, indices = transform(data["events"], np.zeros(2), np.array([-.08, .08]), np.zeros(2, bool), 20.)
    np.testing.assert_array_equal(after[:, 0], [3., 3.])
    np.testing.assert_allclose(after[:, 1], [4.92, 6.08])
    np.testing.assert_array_equal(after[:, 2:], data["events"][:, 2:])
    assert indices.tolist() == [0, 1]


def test_touching_repeated_attacks_move_together_or_abstain():
    events = item([[3., 4., 48, 80], [4., 5., 48, 90]])["events"]
    after, _ = transform(events, np.array([0., .04]), np.array([.04, 0.]), np.zeros(2, bool), 20.)
    assert after[0, 1] == after[1, 0] == 4.04
    after, _ = transform(events, np.array([0., -.08]), np.zeros(2), np.zeros(2, bool), 20.)
    np.testing.assert_array_equal(after, events)


def test_pair_rollback_cannot_introduce_a_collision_earlier_in_a_repeat_chain():
    events = item([[3., 4., 48, 80], [4.02, 4.2, 48, 80], [4.2, 5., 48, 80]])["events"]
    after, _ = transform(events, np.array([0., .04, -.04]), np.array([.04, 0., 0.]), np.zeros(3, bool), 20.)
    assert np.all(after[:-1, 1] <= after[1:, 0])
    assert np.all(np.diff(after[:, 0]) > 0)


def test_edges_and_other_registers_abstain_even_for_confident_removal():
    events = item([[1., 2., 48, 80], [18., 19., 48, 80], [3., 5., 72, 80]])["events"]
    after, indices = transform(events, np.full(3, .08), np.full(3, -.08), np.ones(3, bool), 20.)
    np.testing.assert_array_equal(after, events)
    assert len(indices) == 3


def test_guardian_veto_and_invalid_confidences():
    data = item([[3., 4., 48, 80]])
    ps = heads(1)
    ps[0][:] = [0., 1.]
    assert len(decode(data, ps, np.array([.9]), .9, .9)[0]) == 1
    assert len(decode(data, ps, np.array([.1]), .9, .9)[0]) == 0
    ps[1][0, 0] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        decode(data, ps, np.array([.1]), .9, .9)
    with pytest.raises(ValueError, match="thresholds"):
        decode(data, heads(1), np.array([.1]), -1, .9)


def test_removal_cannot_manufacture_an_onset_or_release_accuracy_gain():
    data = item([[3.04, 4.04, 48, 80], [6., 7., 53, 80]], [[3., 4., 48]])
    ps = heads(2)
    ps[0][1] = [0., 1.]
    report = evaluate([data], [ps], [np.zeros(2)], .9, .9, {})
    assert report["passes"]
    assert report["false_notes_removed"] == 1
    assert report["onset_gain_seconds"] == report["release_gain_seconds"] == 0


def test_shortening_a_true_hold_or_spreading_a_chord_fails_validation():
    data = item([[3., 5., 48, 80], [3., 6., 52, 80]])
    ps = heads(2)
    ps[2][0] = np.eye(7)[0]  # Prematurely release the first chord tone.
    report = evaluate([data], [ps], [np.ones(2)], .9, .9, {})
    assert not report["passes"]
    assert report["per_recording"][0]["coverage"]["lost_reference_seconds"] > 0
    ps = heads(2)
    ps[1][0] = np.eye(7)[6]
    assert not evaluate([data], [ps], [np.ones(2)], .9, .9, {})["passes"]


def test_untrained_model_abstains_and_corrupt_acoustic_windows_fail():
    import torch

    torch.set_num_threads(2)
    data = item([[3., 5., 48, 80]])
    network = model(16)
    normalizer = (np.zeros(100, np.float32), np.ones(100, np.float32))
    ps = probabilities(network, data, normalizer)
    after, _ = decode(data, ps, np.ones(1), .8, .9)
    np.testing.assert_array_equal(after, data["events"])
    assert all(np.isclose(p.sum(), 1) for p in ps)
    assert np.all(SHIFTS[ps[1].argmax(axis=1)] == 0)
    data["release_frames"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="acoustic"):
        probabilities(network, data, normalizer)
