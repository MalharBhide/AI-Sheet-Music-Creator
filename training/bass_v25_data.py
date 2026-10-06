"""Read only the explicit deployed-V25 fitting baseline, never consumed tests."""

import json
from pathlib import Path

import numpy as np
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v25_baseline import VERSION, hashes
from prepare_bass_v25_baseline import COUNTS
from prepare_robust_training_stems import digest


def load(folder, groups=("train", "validation")):
    if any(g not in COUNTS for g in groups):
        raise ValueError("V25 fitting cannot read regressions")
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        plan["version"] != VERSION
        or manifest["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or manifest["plan_sha256"] != digest(folder / "plan.json")
        or plan["producer_sha256"]
        != digest(Path(__file__).with_name("prepare_bass_v25_baseline.py"))
        or any(
            not document[k]
            for document in (plan, manifest)
            for k in ("no_audio_inference", "no_regression_read", "no_user_audio_or_scores")
        )
    ):
        raise ValueError("Changed V25 fitting contract")
    records = manifest["items"]
    partitions = {g: {i["source_group"] for i in records if i["group"] == g} for g in COUNTS}
    if (
        len(records) != 733
        or len({r["id"] for r in records}) != 733
        or {g: sum(i["group"] == g for i in records) for g in COUNTS} != COUNTS
        or partitions["train"] & partitions["validation"]
    ):
        raise ValueError("Incomplete or leaking V25 source groups")
    result = []
    for record in records:
        if record["group"] not in groups:
            continue
        path = folder / "features" / (record["id"] + ".npz")
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed V25 cache or waveform")
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity"])
                != identity({k: v for k, v in record.items() if k != "cache_sha256"})
                or str(saved["plan_sha256"]) != manifest["plan_sha256"]
            ):
                raise ValueError("Changed V25 interval identity")
            item = {**record, **{k: saved[k] for k in ("events", "base_x", "x", "eligible")}}
        n = len(item["events"])
        if (
            n != record["notes"]
            or item["events"].shape != (n, 4)
            or item["base_x"].shape != (n, 52)
            or item["x"].shape != (n, 84)
            or any(not np.isfinite(item[k]).all() for k in ("events", "base_x", "x"))
        ):
            raise ValueError("Invalid V25 feature shape")
        np.testing.assert_array_equal(item["eligible"], eligible(item["events"], item["duration"]))
        np.testing.assert_array_equal(item["x"], features(item["events"], item["base_x"]))
        item.update(
            reference=np.asarray(record["reference"], float).reshape(-1, 3),
            pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3),
        )
        result.append(item)
    return result
