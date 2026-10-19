from pathlib import Path

import numpy as np
import pytest
import torch

import export_approved_finetune_v69 as export
from approved_model_finetune import load_approved
from app.services.left_presence import PresenceModel

ROOT = Path(__file__).resolve().parents[1]


def values():
    approved = load_approved(ROOT)
    winner = {"profile": {"width": 12}, "threshold": .9, "veto_threshold": .2}
    return approved, winner


def test_portable_candidate_preserves_original_float32_weights_and_normalization(tmp_path):
    approved, winner = values()
    data = export.portable_values(approved.network, approved.normalizer, winner)
    path = tmp_path/"portable.npz"
    np.savez_compressed(path, **data)
    with np.load(path, allow_pickle=False) as saved:
        restored = PresenceModel(saved)
    for key, value in approved.network.state_dict().items():
        assert torch.equal(value, restored.network.state_dict()[key])
    for original, actual in zip(approved.normalizer, restored.normalizer, strict=True):
        assert original.dtype == actual.dtype == np.float32
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize("field", ["width", "threshold", "veto_threshold"])
def test_export_cannot_change_approved_thresholds_or_architecture(field):
    approved, winner = values()
    if field == "width":
        winner["profile"][field] = 20
    else:
        winner[field] = .5
    with pytest.raises(ValueError, match="metadata"):
        export.portable_values(approved.network, approved.normalizer, winner)


@pytest.mark.parametrize("fault", ["normalization_dtype", "nonfinite_normalization", "weight_dtype", "nonfinite_weight"])
def test_export_never_silently_casts_or_accepts_damaged_arrays(fault):
    approved, winner = values()
    normalizer = [a.copy() for a in approved.normalizer]
    network = approved.network
    if fault == "normalization_dtype":
        normalizer[0] = normalizer[0].astype(np.float64)
    elif fault == "nonfinite_normalization":
        normalizer[0][0] = float("nan")
    elif fault == "weight_dtype":
        network = network.double()
    else:
        with torch.no_grad():
            next(network.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        export.portable_values(network, normalizer, winner)


def test_failed_release_evidence_cannot_export_or_load_any_parity_data(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "require_export_protocol", lambda *args: "protocol")
    monkeypatch.setattr(export, "frozen", lambda *args: (None, {}, None, None))
    def reject(*args):
        raise ValueError("Fresh proof failed")
    monkeypatch.setattr(export, "require_tests", reject)
    for name in ("fitting_items", "consumed", "storage_preflight", "portable_values"):
        monkeypatch.setattr(export, name, lambda *args: pytest.fail("Must reject before export or parity reads"))
    output = tmp_path/"must-not-exist.npz"
    with pytest.raises(ValueError, match="Fresh"):
        export.export(tmp_path, ROOT, tmp_path, output)
    assert not output.exists()
