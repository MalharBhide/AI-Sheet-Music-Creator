import numpy as np
import pytest
import torch
from attack_release_model import model as previous_model
from attack_release_model import probabilities as previous_probabilities
from fusion_model import initialize, model, probabilities


@pytest.mark.parametrize("width", [32, 64])
def test_new_branch_starts_with_exact_inherited_acoustic_predictions(width):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    previous = previous_model(24)
    with torch.no_grad():
        for parameter in previous.parameters():
            parameter.uniform_(-0.05, 0.05)
    saved = {
        "version": "attack-release-bass-support-v25-v1",
        "width": 24,
        "state_dict": previous.state_dict(),
        "mean": torch.zeros(84),
        "scale": torch.ones(84),
    }
    network = initialize(model(width), saved)
    rng = np.random.default_rng(131)
    item = {
        "events": np.zeros((4, 4)),
        "frames": rng.random((4, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((4, 61, 18), dtype=np.float32),
        "release_frames": rng.random((4, 61, 18), dtype=np.float32),
        "x": rng.random((4, 84), dtype=np.float32),
        "periodicity": rng.uniform(-1, 1, (4, 63)).astype(np.float32),
    }
    expected = previous_probabilities(
        previous,
        item["frames"],
        item["attack_frames"],
        item["release_frames"],
        item["x"],
        (np.zeros(84, np.float32), np.ones(84, np.float32)),
    )
    actual = probabilities(network, item, (np.zeros(147, np.float32), np.ones(147, np.float32)))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    item["periodicity"] *= -1
    np.testing.assert_allclose(
        probabilities(network, item, (np.zeros(147, np.float32), np.ones(147, np.float32))),
        expected,
        atol=1e-6,
    )


def test_changed_warm_architecture_rejected():
    with pytest.raises(ValueError, match="warm-start"):
        initialize(model(32), {"version": "unapproved", "width": 24})


def test_failed_regression_prevents_first_pass_before_creating_output(tmp_path, monkeypatch):
    import fresh_fusion_v27 as producer

    monkeypatch.setattr(producer, "frozen", lambda _: ({}, {}, None, None))

    def refuse(*_):
        raise ValueError("Failed frozen regression")

    monkeypatch.setattr(producer, "regression_gate", refuse)
    output = tmp_path / "never-generated"
    with pytest.raises(ValueError, match="Failed frozen regression"):
        producer.verify(tmp_path, output)
    assert not output.exists()
