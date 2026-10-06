"""Read only instrument-disjoint fitting data with aligned acoustic fusion inputs."""

import json
from pathlib import Path

import numpy as np
from current_bass_v25_baseline import hashes
from periodicity_bass_data import load as parent_items
from prepare_fusion_v27 import VERSION, contracts
from prepare_robust_training_stems import digest


def load(folder):
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        plan["version"] != VERSION
        or manifest["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or plan["code_sha256"] != contracts()
        or manifest["plan_sha256"] != digest(folder / "plan.json")
        or len(manifest["items"]) != 817
        or len({i["id"] for i in manifest["items"]}) != 817
        or not plan["no_regression_audio_or_metrics"]
        or not plan["no_user_audio_or_transcription_inference"]
    ):
        raise ValueError("Changed or leaking fusion fitting observations")
    phase, nsynth = Path(plan["periodicity_root"]), Path(plan["nsynth_root"])
    if (
        digest(phase / "manifest.json") != plan["periodicity_manifest_sha256"]
        or digest(nsynth / "manifest.json") != plan["nsynth_manifest_sha256"]
    ):
        raise ValueError("Changed waveform fusion fitting parents")
    records = {i["id"]: i for i in manifest["items"]}
    items = parent_items(phase, nsynth)
    for item in items:
        record = records.pop(item["id"])
        path = folder / "features" / (item["id"] + ".npz")
        if (
            record["source_group"] != item["source_group"]
            or record["group"] != item["group"]
            or record["parent_cache_sha256"] != item["cache_sha256"]
            or digest(path) != record["cache_sha256"]
        ):
            raise ValueError("Changed fusion candidate/source identity")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed fusion plan identity")
            np.testing.assert_array_equal(saved["events"], item["events"])
            for key, shape in (
                ("frames", (40, 9)),
                ("attack_frames", (61, 18)),
                ("release_frames", (61, 18)),
            ):
                array = saved[key]
                if (
                    array.shape != (len(item["events"]), *shape)
                    or not np.isfinite(array).all()
                    or np.any((array < 0) | (array > 1))
                ):
                    raise ValueError("Invalid aligned acoustic fusion window")
                item[key] = array
    if records:
        raise ValueError("Missing fusion fitting interval")
    return items
