import numpy as np
import pytest
import torch
from musicnet_anchor_model import initialize, model, normalizer, probabilities


@pytest.mark.parametrize("width,cap", [(64, 2.0), (96, 4.0)])
def test_initial_exact_parity_and_full_anchor_frozen_through_training(width, cap):
    from teacher_embedding_model import VERSION
    from teacher_embedding_model import model as approved_model
    from teacher_embedding_model import probabilities as approved_probabilities

    torch.manual_seed(411)
    approved = approved_model(64, 2.0)
    with torch.no_grad():
        # Include nonzero learned V32 correction, not just its acoustic teacher.
        for parameter in approved.parameters():
            parameter.uniform_(-0.04, 0.04)
    saved = dict(
        version=VERSION,
        width=64,
        cap=2.0,
        state_dict=approved.state_dict(),
        mean=torch.zeros(155),
        scale=torch.ones(155),
    )
    candidate = initialize(model(width, cap), saved)
    rng = np.random.default_rng(278)
    item = {
        "events": np.zeros((4, 4)),
        "frames": rng.random((4, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((4, 61, 18), dtype=np.float32),
        "release_frames": rng.random((4, 61, 18), dtype=np.float32),
        "x": rng.normal(0, 1, (4, 84)).astype(np.float32),
        "periodicity": rng.uniform(-1, 1, (4, 63)).astype(np.float32),
        "release_neighbors": rng.uniform(-1, 1, (4, 8)).astype(np.float32),
    }
    stats = normalizer(saved)
    np.testing.assert_array_equal(
        probabilities(candidate, item, stats), approved_probabilities(approved, item, stats)
    )
    state = {k: v.clone() for k, v in candidate.anchor.state_dict().items()}
    tensors = [
        torch.from_numpy(item[k]) for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    tensors.append(
        torch.from_numpy(np.column_stack((item["periodicity"], item["release_neighbors"])))
    )
    optimizer = torch.optim.SGD(candidate.parameters(), lr=0.1)
    for _ in range(3):
        candidate.train()
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(
            candidate(*tensors), torch.tensor([0, 1, 0, 1])
        ).backward()
        optimizer.step()
        assert not candidate.anchor.training and not candidate.anchor.teacher.training
        assert all(p.grad is None and not p.requires_grad for p in candidate.anchor.parameters())
    assert all(torch.equal(v, state[k]) for k, v in candidate.anchor.state_dict().items())
    assert torch.count_nonzero(candidate.correction[-1].weight)


def test_invalid_architecture_and_normalizer_refused():
    with pytest.raises(ValueError, match="architecture"):
        model(48, 2.0)
    with pytest.raises(ValueError, match="normalizer"):
        normalizer(dict(mean=torch.zeros(155), scale=torch.zeros(155)))
