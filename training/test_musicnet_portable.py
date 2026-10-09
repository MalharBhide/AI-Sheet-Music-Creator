"""The optional V33 reader must preserve trained probabilities and eligibility."""

import numpy as np
import pytest
import torch
from app.models import PipelineError
from app.services.bass_embedding_declutter import BassEmbeddingDeclutter
from app.services.bass_musicnet_declutter import BassMusicNetDeclutter, MusicNetSupportModel
from app.services.bass_musicnet_network import VERSION


def asset(network, width, cap):
    values = {
        "version": np.asarray(VERSION),
        "format_version": np.asarray(4),
        "width": np.asarray(width),
        "cap": np.asarray(cap),
        "remove_threshold": np.asarray(0.8),
        "guardian_threshold": np.asarray(0.5),
        "mean": np.zeros(155, np.float32),
        "scale": np.ones(155, np.float32),
    }
    values.update(
        {"weight_" + k: v.detach().numpy().copy() for k, v in network.state_dict().items()}
    )
    return values


def network_and_item(width, cap):
    from musicnet_anchor_model import model

    torch.manual_seed(278)
    network = model(width, cap)
    with torch.no_grad():
        # Exercise a learned nonzero correction as well as all inherited weights.
        for parameter in network.parameters():
            parameter.uniform_(-0.03, 0.03)
    rng = np.random.default_rng(278)
    events = np.asarray([[1, 2, 40, 90], [3, 4, 40, 90], [5, 6, 60, 90], [8, 9, 45, 90]], float)
    item = {
        "events": events,
        "frames": rng.random((4, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((4, 61, 18), dtype=np.float32),
        "release_frames": rng.random((4, 61, 18), dtype=np.float32),
        "x": rng.uniform(-1, 1, (4, 84)).astype(np.float32),
        "periodicity": rng.uniform(-1, 1, (4, 63)).astype(np.float32),
        "release_neighbors": rng.uniform(-1, 1, (4, 8)).astype(np.float32),
    }
    return network, item


@pytest.mark.parametrize("width,cap", [(64, 2.0), (96, 4.0)])
def test_exported_model_matches_trained_model_and_masks_exactly(tmp_path, width, cap):
    from musicnet_anchor_model import probabilities

    network, item = network_and_item(width, cap)
    path = tmp_path / "candidate.npz"
    np.savez_compressed(path, **asset(network, width, cap))
    with np.load(path, allow_pickle=False) as saved:
        portable = MusicNetSupportModel(saved)
    expected = probabilities(network, item, (np.zeros(155, np.float32), np.ones(155, np.float32)))
    actual = portable.probability(item)
    np.testing.assert_array_equal(actual, expected)
    events = item["events"].copy()
    unsupported = np.tile(np.asarray([0.0, 1.0]), (4, 1))
    for stricter in (False, True):
        np.testing.assert_array_equal(
            portable.keep(events, unsupported, np.zeros(4), 10.0, stricter),
            [True, False, True, True],
        )
        np.testing.assert_array_equal(
            portable.keep(events, unsupported, np.ones(4), 10.0, stricter), [True] * 4
        )
    np.testing.assert_array_equal(events, item["events"])


@pytest.mark.parametrize(
    "mutation", ["version", "extra-weight", "wrong-shape", "nonfinite", "normalizer", "threshold"]
)
def test_incompatible_asset_is_refused(tmp_path, mutation):
    network, _ = network_and_item(64, 2.0)
    values = asset(network, 64, 2.0)
    key = "weight_correction.0.weight"
    if mutation == "version":
        values["format_version"] = np.asarray(3)
    elif mutation == "extra-weight":
        values["weight_unexpected"] = np.zeros(1, np.float32)
    elif mutation == "wrong-shape":
        values[key] = values[key][:1]
    elif mutation == "nonfinite":
        values[key][0, 0] = np.nan
    elif mutation == "normalizer":
        values["scale"][0] = 0
    else:
        values["remove_threshold"] = np.asarray(0.1)
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **values)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        MusicNetSupportModel(saved)


def test_optional_stage_reuses_exact_released_physical_filter_and_requires_hash(tmp_path):
    assert BassMusicNetDeclutter.filter is BassEmbeddingDeclutter.filter
    path = tmp_path / "candidate.npz"
    path.write_bytes(b"unverified")
    with pytest.raises(PipelineError, match="damaged"):
        BassMusicNetDeclutter(path, "wrong")
