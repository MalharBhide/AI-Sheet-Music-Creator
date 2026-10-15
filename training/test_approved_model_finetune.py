"""Protect exact released initialization and the user's storage ceiling."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import approved_model_finetune as ft

ROOT = Path(__file__).resolve().parents[1]


def test_loaded_weights_and_normalization_are_exact_deployed_arrays():
    approved = ft.load_approved(ROOT)
    with np.load(ROOT / "backend/app/assets/left-presence-v1.npz", allow_pickle=False) as saved:
        for key, value in approved.network.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), saved["weight_" + key])
        for key, value in zip(("mean", "scale"), approved.normalizer, strict=True):
            np.testing.assert_array_equal(value, saved[key])
            assert not value.flags.writeable
    assert all(p.requires_grad for p in approved.network.parameters())
    assert approved.threshold == .9 and approved.veto_threshold == .2


def test_real_training_step_updates_existing_parameters_without_touching_asset():
    torch.set_num_threads(2)
    approved = ft.load_approved(ROOT)
    before = {k: v.clone() for k, v in approved.network.state_dict().items()}
    asset = ROOT / "backend/app/assets/left-presence-v1.npz"
    sha = ft.sha256(asset)
    generator = torch.Generator().manual_seed(620001)
    inputs = [torch.rand(shape, generator=generator) for shape in
              ((4, 40, 9), (4, 61, 18), (4, 61, 18), (4, 163))]
    optimizer = torch.optim.AdamW(approved.network.parameters(), lr=1e-5)
    approved.network.eval()  # Deterministic unit fixture, not a fitted dataset.
    rejection, presence = approved.network(*inputs)
    loss = torch.nn.functional.cross_entropy(rejection, torch.tensor([0, 0, 1, 1]))
    loss += torch.nn.functional.binary_cross_entropy_with_logits(presence, torch.ones(4, 40))
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in approved.network.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in approved.network.state_dict().items())
    assert ft.sha256(asset) == sha


@pytest.mark.parametrize("field,value", [("status", "unverified"), ("asset_sha256", "0" * 64),
                                        ("new_weights_deployed", False), ("no_user_audio_or_scores", False)])
def test_changed_release_receipt_is_rejected(tmp_path, field, value):
    receipt = json.loads((ROOT / "training/results/left-precision-v43-v1/release-checks.json").read_text())
    receipt[field] = value
    path = tmp_path / "training/results/left-precision-v43-v1/release-checks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="approved deployed"):
        ft.load_approved(tmp_path)


def test_storage_reserves_peak_temporary_and_output_space(tmp_path, monkeypatch):
    (tmp_path / "data").write_bytes(b"x" * 100)
    monkeypatch.setattr(ft.shutil, "disk_usage", lambda _: SimpleNamespace(free=30_000_000_000))
    result = ft.storage_preflight(tmp_path, 1_000_000)
    assert result["used_bytes"] >= 100
    with pytest.raises(ValueError, match="25 GB"):
        ft.storage_preflight(tmp_path, ft.LOCAL_LIMIT_BYTES)


def test_storage_refuses_low_free_space_and_links(tmp_path, monkeypatch):
    monkeypatch.setattr(ft.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(ValueError, match="insufficient"):
        ft.storage_preflight(tmp_path, 1)
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ft.storage_preflight(tmp_path)


@pytest.mark.parametrize("reservation", [-1, 1.5, True])
def test_invalid_reservations_are_rejected(tmp_path, reservation):
    with pytest.raises(ValueError, match="reservation"):
        ft.storage_preflight(tmp_path, reservation)


def test_only_exact_audited_shared_evidence_link_is_allowed(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    evidence = tmp_path / "repository-evidence"
    evidence.mkdir()
    (cache / "results").symlink_to(evidence, target_is_directory=True)
    assert ft.storage_preflight(cache, audited_links={"results": evidence})["used_bytes"] == 0
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="symlink"):
        ft.storage_preflight(cache, audited_links={"results": other})
    with pytest.raises(ValueError, match="Invalid audited"):
        ft.storage_preflight(cache, audited_links={"../outside": evidence})
