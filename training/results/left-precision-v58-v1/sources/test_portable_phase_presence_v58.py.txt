import numpy as np
import pytest
import torch
from app.services.left_presence_network import heads as runtime_heads
from left_precision_v58 import gate, model, probabilities
from portable_phase_presence_v58 import FORMAT, VERSION, AdditionalPhaseModel


def checkpoint(path, model_width=20, **changes):
    torch.manual_seed(193919)
    network = model(model_width).eval()
    # Simulate nontrivial fitted parameters; conservative initial biases alone
    # cannot substantiate a serialized network's numerical parity.
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.uniform_(-.1, .1)
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(FORMAT),
        "width": np.asarray(model_width), "threshold": np.asarray(.98), "veto_threshold": np.asarray(.3),
        "mean": np.linspace(-.3, .3, 163, dtype=np.float32),
        "scale": np.linspace(.1, 1., 163, dtype=np.float32)}
    values.update({"weight_"+name: tensor.numpy().copy() for name, tensor in network.state_dict().items()})
    values.update(changes)
    np.savez_compressed(path, **values)
    return network, (values["mean"], values["scale"])


def item(count=259):
    rng = np.random.default_rng(131781)
    starts = 3.+np.arange(count)*.02
    return {"events": np.column_stack((starts, starts+.15, 42+np.arange(count)%25, np.full(count, 80))),
        "x": rng.random((count, 84), dtype=np.float32),
        "release_neighbors": rng.random((count, 8), dtype=np.float32),
        "frames": rng.random((count, 40, 9), dtype=np.float32),
        "attack_frames": rng.random((count, 61, 18), dtype=np.float32),
        "release_frames": rng.random((count, 61, 18), dtype=np.float32),
        "periodicity": rng.uniform(-1, 1, (count, 63)).astype(np.float32)}


@pytest.mark.parametrize("width", [12, 20])
def test_independent_runtime_and_fitting_probabilities_match_across_batch_boundary(tmp_path, width):
    path = tmp_path/"phase.npz"
    fitted, normalizer = checkpoint(path, width)
    original = item()
    with np.load(path, allow_pickle=False) as saved:
        portable = AdditionalPhaseModel(saved)
    expected = probabilities(fitted, original, normalizer)
    actual_heads = runtime_heads(portable.network, original, portable.normalizer)
    for actual, reference in zip(actual_heads, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    for stricter in (False, True):
        actual = portable.probability(original, stricter)
        np.testing.assert_allclose(actual, gate(expected, .3*(.8 if stricter else 1)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("changes", [
    {"version": np.asarray("other")}, {"format_version": np.asarray(5)},
    {"width": np.asarray(20.1)}, {"threshold": np.asarray(.97)},
    {"veto_threshold": np.asarray(np.nan)}, {"threshold": np.asarray([.98])},
    {"scale": np.zeros(163, np.float32)}, {"scale": np.ones(163, np.float64)},
    {"mean": np.zeros(100, np.float32)}, {"unexpected": np.zeros(1)},
    {"weight_frame.bias": np.zeros(1, np.float64)},
    {"weight_phase": np.zeros((1, 4, 40), np.float32)},
    {"weight_presence.0.weight": np.full((20, 13, 5), np.nan, np.float32)},
])
def test_runtime_rejects_changed_format_clock_weights_or_normalization(tmp_path, changes):
    path = tmp_path/"bad.npz"
    checkpoint(path, **changes)
    with np.load(path, allow_pickle=False) as saved, pytest.raises(ValueError):
        AdditionalPhaseModel(saved)


def test_additional_model_can_never_restore_rejected_baseline_notes_or_move_clocks(tmp_path):
    path = tmp_path/"phase.npz"
    checkpoint(path)
    with np.load(path, allow_pickle=False) as saved:
        portable = AdditionalPhaseModel(saved)
    original = item(5)
    original["events"] = np.array([[1, 2, 48, 80], [3, 4, 72, 80], [3, 4, 48, 80],
                                   [3, 4, 52, 80], [18, 19, 48, 80]], float)
    with torch.no_grad():
        portable.network.frame.weight.zero_()
        portable.network.frame.bias.fill_(-10)
        portable.network.reject.classifier[-1].weight.zero_()
        portable.network.reject.classifier[-1].bias.copy_(torch.tensor([-10., 10.]))
    events = original["events"].copy()
    baseline = np.array([True, False, True, False, True])
    for stricter in (False, True):
        accepted = portable.intersect(baseline, original, 20, stricter)
        np.testing.assert_array_equal(accepted, [True, False, False, False, True])
        assert not np.any(accepted & ~baseline)
        np.testing.assert_array_equal(original["events"], events)
        np.testing.assert_array_equal(original["events"][accepted], events[accepted])
    with pytest.raises(ValueError, match="baseline"):
        portable.intersect(baseline.astype(np.int64), original, 20)


def test_empty_candidate_set_needs_no_network_predictions(tmp_path):
    path = tmp_path/"phase.npz"
    checkpoint(path)
    with np.load(path, allow_pickle=False) as saved:
        portable = AdditionalPhaseModel(saved)
    assert portable.intersect(np.zeros(0, bool), item(0), 20).shape == (0,)
