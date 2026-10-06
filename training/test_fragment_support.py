import numpy as np
import pytest
from fragment_support_model import initialize, model, probabilities, targets


@pytest.mark.parametrize("width", [32, 64])
def test_inherited_keep_mass_survives_three_class_split(width):
    import torch
    from attack_release_model import model as old_model
    from attack_release_model import probabilities as old_probabilities

    torch.set_num_threads(1)
    torch.manual_seed(137)
    old = old_model(24)
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.uniform_(-0.05, 0.05)
    saved = {
        "version": "attack-release-bass-support-v25-v1",
        "width": 24,
        "state_dict": old.state_dict(),
        "mean": torch.zeros(84),
        "scale": torch.ones(84),
    }
    network = initialize(model(width), saved)
    rng = np.random.default_rng(29)
    item = {
        "events": np.zeros((5, 4)),
        "frames": rng.random((5, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((5, 61, 18), dtype=np.float32),
        "release_frames": rng.random((5, 61, 18), dtype=np.float32),
        "x": rng.normal(0, 1, (5, 84)).astype(np.float32),
        "periodicity": rng.uniform(-1, 1, (5, 63)).astype(np.float32),
    }
    expected = old_probabilities(
        old,
        item["frames"],
        item["attack_frames"],
        item["release_frames"],
        item["x"],
        (np.zeros(84), np.ones(84)),
    )
    actual = probabilities(network, item, (np.zeros(147), np.ones(147)))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    item["periodicity"] *= -1
    np.testing.assert_allclose(
        probabilities(network, item, (np.zeros(147), np.ones(147))),
        expected,
        atol=1e-6,
    )


def test_training_taxonomy_preserves_small_continuations_and_duplicate_truth():
    item = {
        "group": "train",
        "events": np.array([[0, 4, 36, 0.8], [0, 4, 40, 0.8], [0, 4, 43, 0.8], [0, 4, 45, 0.8]]),
        "pitch_reference": np.array([[3.993, 4, 36], [3.993, 4, 36], [0, 4, 40]]),
        "y": np.array([1, 1, 0, 1]),
        "mask": np.ones(4, bool),
    }
    # Even 7ms of valid continuation is a protected subtype. A protected attack
    # with no interval overlap remains protected by the original target.
    np.testing.assert_array_equal(targets(item), [0, 1, 2, 1])
    item["pitch_reference"] = np.array([[0, 1, 36], [0, 4, 40]])
    np.testing.assert_array_equal(targets(item), [1, 1, 2, 1])
    item["group"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        targets(item)


def test_failed_regression_refuses_fresh_generation(tmp_path, monkeypatch):
    import fresh_fragment_v28 as producer

    monkeypatch.setattr(producer, "frozen", lambda _: ({}, {}, None, None))

    def reject(*_):
        raise ValueError("Failed frozen regression")

    monkeypatch.setattr(producer, "regression_gate", reject)
    output = tmp_path / "not-generated"
    with pytest.raises(ValueError, match="Failed frozen regression"):
        producer.verify(tmp_path, output)
    assert not output.exists()


def test_float32_protected_sum_cannot_break_strict_probability_bounds(monkeypatch):
    import fragment_support_model as module

    # Real softmax sums can be one ulp above one. Never weaken the decoder's
    # strict probability guard to accommodate a new support taxonomy.
    raw = np.array([[.50000006, .50000006, 1e-9], [0, 0, 1]], np.float32)
    monkeypatch.setattr(module, "binary_probabilities", lambda *_: raw)
    result = probabilities(None, {"events": np.zeros((2, 4))}, None)
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(result.sum(axis=1), 1)
    np.testing.assert_array_equal(result[:, 1], raw[:, 2])
