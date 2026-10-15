import copy

import pytest

import train_approved_finetune_v64 as previous
import train_approved_finetune_v65 as current


def protocol():
    return {"version": current.VERSION, "profiles": list(current.PROFILES), "source_sha256": {"test": "sha"},
            "approved_asset_sha256": current.APPROVED_SHA256, "thresholds": list(current.THRESHOLDS),
            "veto_thresholds": list(current.VETO_THRESHOLDS), "fresh_seeds": list(current.FRESH_SEEDS)}


def test_protocol_keeps_full_dataset_architecture_thresholds_and_preservation_gates():
    assert current.FOLDERS == previous.FOLDERS
    assert current.STAGES == previous.STAGES == (25, 50, 100)
    assert current.THRESHOLDS == previous.THRESHOLDS == (.9,)
    assert current.VETO_THRESHOLDS == previous.VETO_THRESHOLDS == (.2,)
    assert current.initialized is not None and current.CHECKPOINT_SHA256 == previous.CHECKPOINT_SHA256
    assert current.require_validation_proof is previous.require_validation_proof
    assert all(p["freeze_backbone"] is True and p["rehearsal_weight"] > 0 for p in current.PROFILES)
    assert set(previous.SOURCES).issubset(current.SOURCES)
    current.require_protocol(protocol(), {"test": "sha"})


@pytest.mark.parametrize("field", ["version", "profiles", "source_sha256", "approved_asset_sha256",
                                   "thresholds", "veto_thresholds", "fresh_seeds"])
def test_changed_experiment_cannot_start_fitting(field):
    value = copy.deepcopy(protocol())
    value[field] = None
    with pytest.raises(ValueError, match="preregistration"):
        current.require_protocol(value, {"test": "sha"})
