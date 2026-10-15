import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.models import PipelineError
from app.services.left_presence_refinement import (
    APPROVED_SHA256, IntersectedPresence, LeftHandPresenceRefinement,
)

ASSET = Path(__file__).resolve().parents[1]/"app/assets/left-presence-v1.npz"


def fake(decisions, normalizer=None, calls=None):
    if normalizer is None:
        normalizer = (np.zeros(163, np.float32), np.ones(163, np.float32))
    def keep(item, duration, stricter):
        if calls is not None:
            calls.append((id(item), item["events"].copy(), duration, stricter))
        return np.array(decisions)
    return SimpleNamespace(normalizer=normalizer, keep=keep)


@pytest.mark.parametrize("stricter", [False, True])
def test_refinement_cannot_restore_rejected_notes_or_change_observation_context(stricter):
    calls = []
    item = {"events": np.array([[3., 4., 48., 80.], [4., 5., 48., 90.], [4., 8., 52., 70.]])}
    before = item["events"].copy()
    combined = IntersectedPresence(fake([False, True, True], calls=calls),
                                   fake([True, False, True], calls=calls))
    np.testing.assert_array_equal(combined.keep(item, 30., stricter), [False, False, True])
    assert len(calls) == 2
    for identity, events, duration, policy in calls:
        assert identity == id(item) and duration == 30. and policy is stricter
        np.testing.assert_array_equal(events, before)
    np.testing.assert_array_equal(item["events"], before)


@pytest.mark.parametrize("fault", ["mean", "scale", "dtype"])
def test_normalizer_must_remain_identical_to_approved(fault):
    approved = fake([True])
    normalizer = [a.copy() for a in approved.normalizer]
    if fault == "dtype":
        normalizer[0] = normalizer[0].astype(np.float64)
    else:
        normalizer[0 if fault == "mean" else 1][0] += .01
    with pytest.raises(ValueError, match="normalization"):
        IntersectedPresence(approved, fake([True], normalizer))


@pytest.mark.parametrize("decisions", [[1, 0], [True], [[True, False]]])
def test_malformed_masks_fail_closed(decisions):
    combined = IntersectedPresence(fake([True, True]), fake(decisions))
    with pytest.raises(ValueError, match="decisions"):
        combined.keep({"events": np.zeros((2, 4))}, 30.)


def test_empty_candidate_set_keeps_a_valid_empty_mask():
    empty = np.empty(0, bool)
    combined = IntersectedPresence(fake(empty), fake(empty))
    assert combined.keep({"events": np.empty((0, 4))}, 30.).shape == (0,)


def test_original_asset_can_be_loaded_as_an_identity_refinement():
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == APPROVED_SHA256
    verifier = LeftHandPresenceRefinement(ASSET, ASSET, APPROVED_SHA256)
    assert isinstance(verifier.model, IntersectedPresence)
    for key, value in verifier.model.approved.network.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), verifier.model.refinement.network.state_dict()[key].numpy())


def test_damaged_candidate_asset_is_rejected(tmp_path):
    path = tmp_path/"damaged.npz"
    path.write_bytes(b"broken")
    with pytest.raises(PipelineError):
        LeftHandPresenceRefinement(ASSET, path, "changed")


def test_valid_candidate_with_refitted_normalization_is_rejected(tmp_path):
    with np.load(ASSET, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values["mean"][0] += .01
    path = tmp_path/"refitted.npz"
    np.savez_compressed(path, **values)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PipelineError, match="normalization"):
        LeftHandPresenceRefinement(ASSET, path, checksum)
