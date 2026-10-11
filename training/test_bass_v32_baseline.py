import pytest
from bass_v32_data import load
from current_bass_v32_baseline import ASSET_SHA, hashes


def test_historical_v32_arrays_stay_exact_and_old_route_guard_refuses_v38():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "training/results/teacher-embedding-v32-v1/plan.json").read_text())[
        "baseline_hashes"
    ]
    from prepare_robust_training_stems import digest

    for name, expected in {**old["arrays"], "bass-embedding-v1.npz": ASSET_SHA}.items():
        assert digest(root / "backend/app/assets" / name) == expected
    assert digest(root / "backend/app/services/midi_to_score.py") == old["sources"][
        "backend/app/services/midi_to_score.py"]
    # A changed route must not silently reuse caches declared as V32 survivors.
    with pytest.raises(ValueError, match="Changed parity-checked V32"):
        hashes()


def test_legacy_v25_guard_rejects_new_route_without_weakening():
    from current_bass_v25_baseline import hashes as old_hashes

    with pytest.raises(ValueError, match="Changed deployed V25"):
        old_hashes()


@pytest.mark.parametrize("groups", [("test",), ("reserved",), ()])
def test_reserved_and_consumed_groups_refused_before_access(tmp_path, groups):
    with pytest.raises(ValueError, match="reserved"):
        load(tmp_path, groups)
