import numpy as np
import pytest
import torch
from musicnet_full_encoding_model import initialize, model, normalizer, probabilities


@pytest.mark.parametrize("width,cap", [(128, 4.0), (192, 6.0)])
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
    assert candidate.correction[0].in_features == 955
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
        probabilities(candidate, item, stats),
        approved_probabilities(approved, item, stats),
    )
    state = {k: v.clone() for k, v in candidate.anchor.state_dict().items()}
    tensors = [
        torch.from_numpy(item[k])
        for k in ("frames", "attack_frames", "release_frames", "x")
    ]
    tensors.append(
        torch.from_numpy(
            np.column_stack((item["periodicity"], item["release_neighbors"]))
        )
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
        assert all(
            p.grad is None and not p.requires_grad
            for p in candidate.anchor.parameters()
        )
    assert all(
        torch.equal(v, state[k]) for k, v in candidate.anchor.state_dict().items()
    )
    assert torch.count_nonzero(candidate.correction[-1].weight)


def test_invalid_architecture_and_normalizer_refused():
    with pytest.raises(ValueError, match="architecture"):
        model(48, 2.0)
    with pytest.raises(ValueError, match="normalizer"):
        normalizer(dict(mean=torch.zeros(155), scale=torch.zeros(155)))


def test_branch_receives_each_frozen_time_position_and_signed_context():
    from teacher_embedding_model import VERSION
    from teacher_embedding_model import model as approved_model

    torch.manual_seed(279)
    approved = approved_model(64, 2.0)
    saved = dict(
        version=VERSION,
        width=64,
        cap=2.0,
        state_dict=approved.state_dict(),
        mean=torch.zeros(155),
        scale=torch.ones(155),
    )
    candidate = initialize(model(128, 4.0), saved)
    args = [
        torch.rand(2, *shape) for shape in ((40, 9), (61, 18), (61, 18), (84,), (71,))
    ]
    args[3] -= 0.5
    args[4] -= 0.5
    captured = []
    handle = candidate.correction[0].register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach().clone())
    )
    with torch.inference_mode():
        candidate(*args)
        teacher = candidate.anchor.teacher
        coarse = teacher.coarse(args[0].transpose(1, 2))
        attack = teacher.local(args[1].transpose(1, 2))
        release = teacher.ending(args[2].transpose(1, 2))
        hidden = teacher.classifier[:-1](
            torch.cat((coarse, args[3], attack, release), dim=1)
        )
    handle.remove()
    assert coarse.shape == (2, 384) and attack.shape == release.shape == (2, 192)
    assert captured[0].shape == (2, 955)
    torch.testing.assert_close(captured[0][:, :71], args[4], rtol=0, atol=0)
    torch.testing.assert_close(captured[0][:, 71:155], args[3], rtol=0, atol=0)
    torch.testing.assert_close(
        captured[0][:, 155:],
        torch.log1p(torch.cat((coarse, attack, release, hidden), dim=1)),
        rtol=0,
        atol=0,
    )
