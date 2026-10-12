import acquire_nsynth_dynamics_v44 as acquisition
import numpy as np
import pytest
from acquire_nsynth_subset import select
from left_precision_v44 import evaluate
from prepare_nsynth_dynamics_v44 import labels


def test_new_velocities_retain_parent_instrument_partitions_and_exclude_reserve(monkeypatch):
    metadata = {}
    for family, count in (("bass", 10), ("keyboard", 8)):
        for index in range(count):
            instrument = f"{family}_acoustic_{index:03d}"
            for pitch in range(21, 60):
                for velocity in (25, 50, 75, 100, 127):
                    key = f"{instrument}-{pitch:03d}-{velocity:03d}"
                    metadata[key] = {"instrument_family_str": family, "instrument_str": instrument,
                                     "pitch": pitch, "velocity": velocity}
    metadata.update({f"ignored-{i}": {"instrument_family_str": "string", "instrument_str": "string_000",
                                    "pitch": 80, "velocity": 75} for i in range(12678-len(metadata))})
    monkeypatch.setattr(acquisition, "EXPECTED", {"train": 1716, "validation": 468})
    previous = select(metadata)
    groups = {n["instrument_str"]: n["group"] for n in previous.values()}
    chosen = acquisition.selection(metadata)
    assert len(chosen) == 2184
    assert {n["velocity"] for n in chosen.values()} == {25, 50, 100, 127}
    assert all(n["group"] == groups[n["instrument_str"]] != "reserved" for n in chosen.values())
    assert len({n["instrument_str"] for n in chosen.values()}) == 14


def test_authored_releases_protect_the_complete_decay_and_preserve_repeated_spacing():
    notes = [{"id": f"bass_000-{pitch:03d}-025", "pitch": pitch, "instrument_str": "bass_000",
              "group": "train"} for pitch in (24, 36, 48, 55)]
    reference, pitch, _ = labels(notes, "long-overlap")
    np.testing.assert_allclose(np.asarray(reference)[:, 1]-np.asarray(reference)[:, 0], 3.)
    np.testing.assert_allclose(np.asarray(pitch)[:, 1]-np.asarray(pitch)[:, 0], 4.)
    repeated, support, _ = labels(notes, "variable-repeated")
    for index, step in enumerate((.45, .9, .6, 1.2)):
        phrase = np.asarray(repeated[index*3:index*3+3])
        np.testing.assert_allclose(np.diff(phrase[:, 0]), step)
        assert len(set(phrase[:, 2])) == 1
    assert all(tail[1] > key[1] for key, tail in zip(repeated, support, strict=True))


def score_fixture(old_keep):
    full = np.array([[3., 4., 40., 80.], [5., 6., 47., 80.]])
    original = {"id": "fixture", "events": full, "duration": 20., "corpus": "musicnet-real-piano",
                "reference": np.array([[3., 4., 40.]]), "pitch_reference": np.array([[3., 4., 40.]])}
    old_keep = np.array(old_keep)
    deployed = {**original, "events": full[old_keep], "full_release_events": full,
                "baseline_accepted": old_keep, "baseline_policy": "raw"}
    return original, deployed


def test_restoring_an_extra_note_removed_by_live_v43_is_a_regression():
    original, deployed = score_fixture([True, False])
    result = evaluate([original], [deployed], [np.array([[1., 0.], [1., 0.]])], .9)
    assert not result["passes"]
    assert result["false_notes_removed"] == -1
    assert result["recovered_notes"] == 1


def test_removing_a_real_note_fails_even_if_false_note_count_does_not_increase():
    original, deployed = score_fixture([True, False])
    result = evaluate([original], [deployed], [np.array([[0., 1.], [0., 1.]])], .9)
    assert not result["passes"]
    assert result["per_recording"][0]["matches"]["hold"]["lost"] == 1


def test_new_extra_note_rejection_cannot_claim_a_timing_improvement():
    original, deployed = score_fixture([True, True])
    result = evaluate([original], [deployed], [np.array([[1., 0.], [0., 1.]])], .9)
    assert result["passes"] and result["false_notes_removed"] == result["real_piano_gain"] == 1
    row = result["per_recording"][0]
    assert row["retained_clocks_identical"]
    assert row["timing"]["before_onset_release_error_sum"] == row["timing"]["after_onset_release_error_sum"]
    changed = {**deployed, "full_release_events": deployed["full_release_events"].copy()}
    changed["full_release_events"][0, 1] += .1
    with pytest.raises(ValueError, match="release context"):
        evaluate([original], [changed], [np.array([[1., 0.], [0., 1.]])], .9)


def test_reference_matched_predictions_cannot_be_replaced_by_nearby_notes():
    original, _ = score_fixture([True, True])
    full = np.array([[3., 4., 40., 80.], [2.99, 4.02, 40., 80.]])
    original = {**original, "events": full}
    deployed = {**original, "full_release_events": full, "baseline_accepted": np.array([True, True]),
                "baseline_policy": "raw"}
    from train_melody_timing import matches

    matched = matches(original["reference"], full)[0][1]
    probability = np.tile([1., 0.], (2, 1))
    probability[matched] = [0., 1.]
    result = evaluate([original], [deployed], [probability], .9)
    assert not result["passes"]
    assert not result["per_recording"][0]["fixed_reference_identities_preserved"]
