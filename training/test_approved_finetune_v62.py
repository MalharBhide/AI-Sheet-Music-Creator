"""The new training run resumes the approved model, not a failed experiment."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import train_approved_finetune_v62 as run
from approved_model_finetune import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return {"version": run.VERSION, "profiles": deepcopy(list(run.PROFILES)),
            "source_sha256": {"example": "123"}, "approved_asset_sha256": run.APPROVED_SHA256,
            "thresholds": [.9], "veto_thresholds": [.2], "fresh_seeds": list(range(305101, 305141))}


def test_decision_thresholds_and_existing_architecture_are_fixed():
    assert run.THRESHOLDS == (.9,) and run.VETO_THRESHOLDS == (.2,)
    assert all(p["width"] == 12 and p["learning_rate"] <= 1e-5 for p in run.PROFILES)
    assert len(run.PROFILES) == 2
    run.require_protocol(protocol(), {"example": "123"})


@pytest.mark.parametrize("field", ["version", "profiles", "source_sha256", "approved_asset_sha256",
                                   "thresholds", "veto_thresholds", "fresh_seeds"])
def test_protocol_changes_cannot_start_fitting(field):
    record = protocol()
    record[field] = None
    with pytest.raises(ValueError, match="preregistration"):
        run.require_protocol(record, {"example": "123"})


def test_every_profile_starts_from_an_independent_exact_approved_checkpoint(tmp_path, monkeypatch):
    approved = load_approved(ROOT)
    checkpoint = tmp_path / "approved.pt"
    torch.save({"version": "calibrated-frame-support-v43-v1", "width": 12, "epoch": 50,
                "state_dict": approved.network.state_dict(),
                "mean": torch.from_numpy(approved.normalizer[0].copy()),
                "scale": torch.from_numpy(approved.normalizer[1].copy())}, checkpoint)
    monkeypatch.setattr(run, "CHECKPOINT_SHA256", sha256(checkpoint))
    first, first_normalizer, proof = run.initialized(ROOT, checkpoint)
    second, second_normalizer, _ = run.initialized(ROOT, checkpoint)
    assert proof["all_weights_and_normalization_identical"]
    assert first is not second
    with torch.no_grad():
        next(first.parameters()).add_(1)
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
    for a,b in zip(first_normalizer,second_normalizer,strict=True):
        np.testing.assert_array_equal(a,b)
        assert not a.flags.writeable and not b.flags.writeable
