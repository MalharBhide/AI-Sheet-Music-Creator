import numpy as np
import pytest
import torch
from release_neighbor_model import initialize, model, probabilities


@pytest.mark.parametrize("width", [32, 64])
def test_new_release_branch_inherits_exact_shipped_probabilities(width):
    from attack_release_model import model as previous_model
    from attack_release_model import probabilities as previous_probabilities

    torch.set_num_threads(1)
    torch.manual_seed(37)
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
    rng = np.random.default_rng(77)
    item = {
        "events": np.zeros((5, 4)),
        "frames": rng.random((5, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((5, 61, 18), dtype=np.float32),
        "release_frames": rng.random((5, 61, 18), dtype=np.float32),
        "x": rng.normal(0, 1, (5, 84)).astype(np.float32),
        "periodicity": rng.uniform(-1, 1, (5, 63)).astype(np.float32),
        "release_neighbors": rng.uniform(-1, 1, (5, 8)).astype(np.float32),
    }
    expected = previous_probabilities(
        previous,
        item["frames"],
        item["attack_frames"],
        item["release_frames"],
        item["x"],
        (np.zeros(84), np.ones(84)),
    )
    actual = probabilities(network, item, (np.zeros(155), np.ones(155)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.all((actual >= 0) & (actual <= 1))
    item["release_neighbors"] *= -1
    np.testing.assert_allclose(
        probabilities(network, item, (np.zeros(155), np.ones(155))), expected, atol=1e-6
    )
    item["release_neighbors"] = np.zeros((4, 8))
    with pytest.raises(ValueError, match="alignment"):
        probabilities(network, item, (np.zeros(155), np.ones(155)))


def test_reserved_fitting_request_refused_before_any_file_access(tmp_path):
    from groove_bass_data import load

    with pytest.raises(ValueError, match="reserved"):
        load(tmp_path / "missing-baseline", tmp_path / "missing-mixtures", ("reserved",))


def test_failed_regression_stops_new_reserved_sources(tmp_path, monkeypatch):
    import fresh_release_neighbors_v29 as producer

    monkeypatch.setattr(producer, "frozen", lambda _: ({}, {}, None, None))

    def reject(*_):
        raise ValueError("Failed frozen regression")

    monkeypatch.setattr(producer, "regression_gate", reject)
    with pytest.raises(ValueError, match="Failed frozen"):
        producer.verify(tmp_path, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_first_pass_uses_exact_checksum_keys_and_complete_regression_ids(tmp_path, monkeypatch):
    import json

    import fresh_release_neighbors_v29 as producer
    from prepare_robust_training_stems import digest

    observations = tmp_path / "observations"
    observations.mkdir()
    ids = [f"consumed-{i}" for i in range(392)]
    (observations / "manifest.json").write_text(json.dumps({"items": [{"id": i} for i in ids]}))
    winner = {"threshold": 0.7, "guardian_threshold": 0.5}
    monkeypatch.setattr(producer, "require_regression", lambda _: winner)
    monkeypatch.setattr(
        producer,
        "frozen",
        lambda _: ({"periodicity_observation_root": str(observations)}, winner, None, None),
    )
    report = {"observation_manifest_sha256": digest(observations / "manifest.json")}
    for name, threshold in (("raw", 0.7), ("margin", 0.85)):
        report[name] = {
            "passes": True,
            "false_notes_removed": 1,
            "per_recording": [{"id": i} for i in ids],
            "remove_threshold": threshold,
            "guardian_threshold": 0.5,
            "retimed_notes": 0,
            "onset_error_reduction_seconds": 0,
        }
    path = tmp_path / "consumed-regression.json"
    path.write_text(json.dumps(report))
    assert producer.regression_gate(tmp_path, winner) == digest(path)
    report["margin"]["per_recording"] = report["margin"]["per_recording"][:-1]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration"):
        producer.regression_gate(tmp_path, winner)
