import hashlib

import numpy as np
import pytest
import torch
from app.models import PipelineError
from app.services.bass_embedding_declutter import BassEmbeddingDeclutter, EmbeddingSupportModel
from app.services.bass_embedding_network import VERSION, model


def asset(path, **changes):
    network = model(64, 2.0)
    values = dict(
        version=np.asarray(VERSION),
        format_version=np.asarray(3),
        width=np.asarray(64),
        cap=np.asarray(2.0),
        remove_threshold=np.asarray(0.8),
        guardian_threshold=np.asarray(0.5),
        mean=np.zeros(155, np.float32),
        scale=np.ones(155, np.float32),
    )
    values.update({"weight_" + k: v.numpy() for k, v in network.state_dict().items()})
    values.update(changes)
    np.savez_compressed(path, **values)
    return values


@pytest.mark.parametrize(
    "changes",
    [
        {"cap": np.asarray(20.0)},
        {"scale": np.zeros(155, np.float32)},
        {"mean": np.zeros(84, np.float32)},
        {"remove_threshold": np.asarray(float("nan"))},
    ],
)
def test_invalid_portable_contract_refused(tmp_path, changes):
    path = tmp_path / "invalid.npz"
    asset(path, **changes)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        EmbeddingSupportModel(saved)


def test_tampered_asset_refused_before_model_load(tmp_path):
    path = tmp_path / "asset.npz"
    asset(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(PipelineError, match="damaged"):
        BassEmbeddingDeclutter(path, checksum)


def test_boundaries_strong_acoustic_evidence_and_retained_clocks_protected(tmp_path):
    path = tmp_path / "asset.npz"
    asset(path)
    with np.load(path, allow_pickle=False) as saved:
        portable = EmbeddingSupportModel(saved)
    # Includes near-boundary attacks/releases, strong acoustically supported
    # notes, and an upper-register melody note: all must remain unchanged.
    events = np.array(
        [[1, 2, 48, 80], [3, 4, 48, 80], [4, 5, 48, 80], [5, 6, 72, 80], [8, 9, 48, 80]], float
    )
    original = events.copy()
    probabilities = np.tile([0.01, 0.99], (5, 1))
    guards = np.array([0, 0, 0.8, 0, 0])
    keep = portable.keep(events, probabilities, guards, 10)
    np.testing.assert_array_equal(keep, [True, False, True, True, True])
    np.testing.assert_array_equal(events, original)
    assert all(not p.requires_grad for p in portable.network.teacher.parameters())


def test_runtime_initial_logits_equal_acoustic_teacher(tmp_path):
    path = tmp_path / "asset.npz"
    asset(path)
    with np.load(path, allow_pickle=False) as saved:
        portable = EmbeddingSupportModel(saved)
    args = [torch.zeros(2, *shape) for shape in ((40, 9), (61, 18), (61, 18), (84,))]
    with torch.inference_mode():
        expected = portable.network.teacher(*args)
        actual = portable.network(*args, torch.zeros(2, 71))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
