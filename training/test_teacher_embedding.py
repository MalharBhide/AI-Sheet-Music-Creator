import numpy as np
import pytest
import torch
from teacher_embedding_model import initialize, model, probabilities


def setup(width, cap):
    from attack_release_model import model as previous_model

    torch.manual_seed(17)
    previous = previous_model(24)
    with torch.no_grad():
        for parameter in previous.parameters():
            parameter.uniform_(-0.04, 0.04)
    saved = {
        "version": "attack-release-bass-support-v25-v1",
        "width": 24,
        "state_dict": previous.state_dict(),
        "mean": torch.zeros(84),
        "scale": torch.ones(84),
    }
    network = initialize(model(width, cap), saved)
    rng = np.random.default_rng(81)
    item = {
        "events": np.zeros((4, 4)),
        "frames": rng.random((4, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((4, 61, 18), dtype=np.float32),
        "release_frames": rng.random((4, 61, 18), dtype=np.float32),
        "x": rng.normal(0, 1, (4, 84)).astype(np.float32),
        "periodicity": rng.uniform(-1, 1, (4, 63)).astype(np.float32),
        "release_neighbors": rng.uniform(-1, 1, (4, 8)).astype(np.float32),
    }
    return previous, network, item


@pytest.mark.parametrize("width,cap", [(64, 2.0), (96, 4.0)])
def test_initial_exact_parity_and_frozen_teacher_after_optimizer_steps(width, cap):
    from attack_release_model import probabilities as previous_probabilities

    previous, network, item = setup(width, cap)
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
    state = {k: v.clone() for k, v in network.teacher.state_dict().items()}
    tensors = [
        torch.from_numpy(item[k]) for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    tensors.append(
        torch.from_numpy(np.column_stack((item["periodicity"], item["release_neighbors"])))
    )
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    for _ in range(3):
        network.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(network(*tensors), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        optimizer.step()
        assert not network.teacher.training
        assert all(p.grad is None and not p.requires_grad for p in network.teacher.parameters())
    assert all(torch.equal(v, state[k]) for k, v in network.teacher.state_dict().items())
    assert torch.count_nonzero(network.correction[-1].weight)


@pytest.mark.parametrize("width,cap", [(64, 2.0), (96, 4.0)])
def test_correction_cannot_exceed_declared_logit_bound(width, cap):
    previous, network, item = setup(width, cap)
    previous.eval()
    tensors = [
        torch.from_numpy(item[k]) for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    phase = torch.from_numpy(np.column_stack((item["periodicity"], item["release_neighbors"])))
    with torch.no_grad():
        baseline = previous(*tensors)
        for sign in (-1, 1):
            network.correction[-1].weight.fill_(0)
            network.correction[-1].bias.fill_(sign * 1000)
            candidate = network(*tensors, phase)
            torch.testing.assert_close(candidate[:, 0], baseline[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(
                candidate[:, 1] - baseline[:, 1], torch.full((4,), sign * cap), atol=1e-6, rtol=0
            )


def test_unapproved_caps_refused():
    with pytest.raises(ValueError, match="architecture"):
        model(64, 10)


def test_new_branch_receives_spectral_representation():
    _, network, item = setup(64, 2.0)
    assert network.correction[0].in_features == 103
    tensors = [
        torch.from_numpy(item[k]) for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    phase = torch.zeros(4, 71)
    observed = []
    hook = network.correction[0].register_forward_pre_hook(
        lambda _, args: observed.append(args[0].detach().clone())
    )
    try:
        network(*tensors, phase)
    finally:
        hook.remove()
    assert observed[0].shape == (4, 103)
    torch.testing.assert_close(observed[0][:, :71], phase)
    assert torch.all(observed[0][:, 71:] >= 0)
    assert not torch.equal(observed[0][0, 71:], observed[0][1, 71:])


def test_failed_regression_stops_first_pass(tmp_path, monkeypatch):
    import fresh_teacher_embedding_v32 as producer

    monkeypatch.setattr(producer, "frozen", lambda _: ({}, {}, None, None))

    def refuse(*_):
        raise ValueError("Failed frozen regression")

    monkeypatch.setattr(producer, "regression_gate", refuse)
    with pytest.raises(ValueError, match="Failed frozen"):
        producer.verify(tmp_path, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
