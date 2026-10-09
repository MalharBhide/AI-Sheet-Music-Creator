"""Read complete consumed regression evidence only after candidate freezing."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v32_baseline import VERSION, hashes
from periodicity_bass_data import validate
from prepare_robust_training_stems import digest
from release_neighbor_features import features as neighbors


def load(folder, expected_manifest_sha256):
    if digest(folder / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Changed frozen consumed regression manifest")
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        plan["version"] != VERSION
        or manifest["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or manifest["plan_sha256"] != digest(folder / "plan.json")
        or plan["producer_sha256"]
        != digest(Path(__file__).with_name("prepare_bass_v32_regression.py"))
        or plan["required_regression_recordings"] != 469
        or not plan["no_decoder_inference"]
        or not plan["no_candidate_or_outcome_metrics"]
        or not plan["no_fitting_selection_user_audio_or_scores"]
        or not manifest["no_fitting_selection_user_audio_or_scores"]
        or not manifest["all_rows_consumed_regression"]
    ):
        raise ValueError("Changed consumed V32 regression contract")
    rows = manifest["items"]
    if (
        len(rows) != 469
        or len({r["id"] for r in rows}) != 469
        or any(r["group"] != "test" for r in rows)
    ):
        raise ValueError("Incomplete consumed V32 regression coverage")
    result = []
    for record in rows:
        path = folder / "features" / (record["id"] + ".npz")
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed consumed regression cache or waveform")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"] or str(
                saved["identity"]
            ) != identity({k: v for k, v in record.items() if k != "cache_sha256"}):
                raise ValueError("Changed consumed regression interval identity")
            item = {
                **record,
                **{
                    k: saved[k]
                    for k in (
                        "events",
                        "base_x",
                        "x",
                        "eligible",
                        "frames",
                        "attack_frames",
                        "release_frames",
                        "periodicity",
                        "release_neighbors",
                    )
                },
            }
        if len(item["events"]) != record["notes"]:
            raise ValueError("Changed consumed V32 survivor count")
        for key, shape in (
            ("frames", (40, 9)),
            ("attack_frames", (61, 18)),
            ("release_frames", (61, 18)),
        ):
            array = item[key]
            if (
                array.shape != (len(item["events"]), *shape)
                or not np.isfinite(array).all()
                or np.any((array < 0) | (array > 1))
            ):
                raise ValueError("Invalid consumed acoustic observation windows")
        np.testing.assert_array_equal(
            item["eligible"], eligible(item["events"], record["duration"])
        )
        np.testing.assert_array_equal(item["x"], features(item["events"], item["base_x"]))
        np.testing.assert_array_equal(
            item["release_neighbors"], neighbors(item["events"], item["base_x"])
        )
        item.update(
            reference=np.asarray(record["reference"], float).reshape(-1, 3),
            pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3),
        )
        result.append(validate(item))
    return result
