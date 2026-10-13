"""Read pinned deployed-V32 fitting survivors; forbid consumed and reserved tests."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v32_baseline import VERSION, hashes
from periodicity_bass_data import validate
from prepare_bass_v32_baseline import COUNTS
from prepare_robust_training_stems import digest
from release_neighbor_features import features as neighbors


def load(folder, groups=("train", "validation")):
    if not groups or any(g not in COUNTS for g in groups):
        raise ValueError("V32 fitting cannot read reserved or consumed regressions")
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        plan["version"] != VERSION
        or manifest["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or manifest["plan_sha256"] != digest(folder / "plan.json")
        or plan["producer_sha256"]
        != digest(Path(__file__).with_name("prepare_bass_v32_baseline.py"))
        or any(
            not doc[k]
            for doc in (plan, manifest)
            for k in ("no_audio_inference", "no_regression_read", "no_user_audio_or_scores")
        )
    ):
        raise ValueError("Changed V32 fitting contract")
    for key in ("baseline", "mixtures"):
        if digest(Path(plan[key + "_root"]) / "manifest.json") != plan[key + "_manifest_sha256"]:
            raise ValueError("Changed guarded V25 fitting parent")
    records = manifest["items"]
    partitions = {g: {r["source_group"] for r in records if r["group"] == g} for g in COUNTS}
    if (
        len(records) != 908
        or len({r["id"] for r in records}) != 908
        or {g: sum(r["group"] == g for r in records) for g in COUNTS} != COUNTS
        or partitions["train"] & partitions["validation"]
    ):
        raise ValueError("Incomplete or leaking V32 source partition")
    result = []
    for record in records:
        if record["group"] not in groups:
            continue
        path = folder / "features" / (record["id"] + ".npz")
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed V32 observation or physical waveform")
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity"])
                != identity({k: v for k, v in record.items() if k != "cache_sha256"})
                or str(saved["plan_sha256"]) != manifest["plan_sha256"]
            ):
                raise ValueError("Changed V32 interval identity")
            item = {
                **record,
                **{
                    k: saved[k]
                    for k in (
                        "events",
                        "base_x",
                        "x",
                        "periodicity",
                        "release_neighbors",
                        "eligible",
                        "frames",
                        "attack_frames",
                        "release_frames",
                    )
                },
            }
        if len(item["events"]) != record["notes"]:
            raise ValueError("Changed V32 survivor count")
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
                raise ValueError("Invalid V32 acoustic observation windows")
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
