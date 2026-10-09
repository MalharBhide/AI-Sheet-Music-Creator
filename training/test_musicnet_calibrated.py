import copy

import numpy as np
import pytest
import torch
from musicnet_calibrated_guardian_model import initialize, model, probabilities
from score_musicnet_calibrated import choose
from teacher_embedding_model import VERSION as ANCHOR_VERSION
from teacher_embedding_model import model as anchor_model
from teacher_embedding_model import probabilities as anchor_probabilities


def saved_anchor():
    torch.manual_seed(45)
    anchor = anchor_model(64, 2.0)
    with torch.no_grad():
        anchor.correction[-1].weight.fill_(0.01)
        anchor.correction[-1].bias.fill_(0.2)
    return anchor, {
        "version": ANCHOR_VERSION,
        "width": 64,
        "cap": 2.0,
        "state_dict": anchor.state_dict(),
        "mean": torch.zeros(155),
        "scale": torch.ones(155),
    }


def observation():
    return {
        "events": np.array([[3.0, 4.0, 48.0, 80.0], [5.0, 6.0, 50.0, 70.0]]),
        "frames": np.zeros((2, 40, 9), np.float32),
        "attack_frames": np.zeros((2, 61, 18), np.float32),
        "release_frames": np.zeros((2, 61, 18), np.float32),
        "x": np.zeros((2, 84), np.float32),
        "periodicity": np.zeros((2, 63), np.float32),
        "release_neighbors": np.zeros((2, 8), np.float32),
    }


def test_initial_removal_matches_complete_anchor_and_protection_starts_closed():
    anchor, saved = saved_anchor()
    network = initialize(model(128, 4.0), saved)
    norm = np.zeros(155, np.float32), np.ones(155, np.float32)
    item = observation()
    removal, protection = probabilities(network, item, norm)
    np.testing.assert_array_equal(removal, anchor_probabilities(anchor, item, norm))
    np.testing.assert_allclose(protection, 0.99, atol=1e-7)
    assert np.all(protection >= 0.5)
    item["x"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid acoustic"):
        probabilities(network, item, norm)


def test_training_both_heads_cannot_change_approved_weights():
    _, saved = saved_anchor()
    network = initialize(model(192, 6.0), saved)
    original = {k: v.clone() for k, v in network.anchor.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=0.01)
    item = observation()
    args = [torch.from_numpy(item[k]) for k in ("frames", "attack_frames", "release_frames", "x")]
    args.append(torch.zeros(2, 71))
    for _ in range(3):
        optimizer.zero_grad()
        removal, protection = network.both(*args)
        (
            torch.nn.functional.cross_entropy(removal, torch.tensor([0, 1]))
            + torch.nn.functional.binary_cross_entropy_with_logits(
                protection, torch.tensor([1.0, 0.0])
            )
        ).backward()
        optimizer.step()
    assert all(torch.equal(v, original[k]) for k, v in network.anchor.state_dict().items())
    assert all(not p.requires_grad for p in network.anchor.parameters())
    assert not network.anchor.training
    assert not torch.equal(
        network.correction[-1].weight, torch.zeros_like(network.correction[-1].weight)
    )
    assert not torch.equal(
        network.protection[-1].weight, torch.zeros_like(network.protection[-1].weight)
    )


def selection_fixture():
    items = [
        {"id": f"real-{i}", "corpus": "musicnet-real-piano", "group": "validation"}
        for i in range(15)
    ]
    rows = [
        {
            "id": i["id"],
            "passes": True,
            "baseline": {"false_positives": 2},
            "candidate": {"false_positives": 2},
        }
        for i in items
    ]
    report = {
        "passes": True,
        "false_notes_removed": 10,
        "failed_recordings": [],
        "per_recording": rows,
    }
    return items, report


def test_authored_only_gain_does_not_select_a_calibrated_winner(monkeypatch):
    items, report = selection_fixture()
    monkeypatch.setattr(
        "score_musicnet_calibrated.evaluate", lambda *args, **kwargs: copy.deepcopy(report)
    )
    selected, search = choose(items, [], [], [0.85], [0.3], {})
    assert selected is None and not search[0]["eligible"]
    report["per_recording"][0]["candidate"]["false_positives"] = 1
    selected, search = choose(items, [], [], [0.85], [0.3], {})
    assert selected["real_piano_gain"] == selected["margin_real_piano_gain"] == 1
    report["per_recording"][-1]["passes"] = False
    assert choose(items, [], [], [0.85], [0.3], {})[0] is None


def test_normal_only_real_gain_and_incomplete_partitions_are_refused(monkeypatch):
    items, report = selection_fixture()

    def mocked(_items, _ps, _guards, threshold, *args):
        result = copy.deepcopy(report)
        if threshold < 0.9:
            result["per_recording"][0]["candidate"]["false_positives"] = 1
        return result

    monkeypatch.setattr("score_musicnet_calibrated.evaluate", mocked)
    assert choose(items, [], [], [0.85], [0.3], {})[0] is None
    with pytest.raises(ValueError, match="exactly 15"):
        choose(items[:-1], [], [], [0.85], [0.3], {})


def test_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="architecture"):
        model(64, 4.0)
