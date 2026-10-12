"""Join sealed V25 intervals and waveform recurrence; forbid reserved fitting."""

import json
from pathlib import Path

import numpy as np
from bass_temporal_data import annotate
from bass_v25_data import load as parent_items
from current_bass_v25_baseline import hashes
from periodicity_features import NAMES, VERSION
from prepare_nsynth_bass import contracts as nsynth_contracts
from prepare_periodicity_bass import contracts
from prepare_robust_training_stems import digest


def validate(item):
    events, x, base, phase = [item[k] for k in ("events", "x", "base_x", "periodicity")]
    n = len(events)
    if (
        events.shape != (n, 4)
        or x.shape != (n, 84)
        or base.shape != (n, 52)
        or phase.shape != (n, len(NAMES))
        or any(not np.isfinite(v).all() for v in (events, x, base, phase))
        or np.any(np.abs(phase) > 1)
    ):
        raise ValueError("Invalid candidate-aligned periodicity features")
    return annotate(item)


def load(folder, nsynth, groups=("train", "validation")):
    if not groups or any(g not in ("train", "validation") for g in groups):
        raise ValueError("Periodicity fitting cannot read consumed or reserved tests")
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        plan["version"] != VERSION
        or manifest["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or plan["code_sha256"] != contracts()
        or manifest["plan_sha256"] != digest(folder / "plan.json")
        or len(manifest["items"]) != 733
        or len({i["id"] for i in manifest["items"]}) != 733
        or not plan["no_regression_audio_or_metrics"]
        or not plan["no_inference_or_user_audio"]
    ):
        raise ValueError("Changed periodicity fitting observations")
    parent = Path(plan["parent_root"])
    if (
        digest(parent / "manifest.json") != plan["parent_manifest_sha256"]
        or digest(parent / "plan.json") != plan["parent_plan_sha256"]
    ):
        raise ValueError("Changed V25 fitting parent")
    records = {i["id"]: i for i in manifest["items"]}
    result = []
    for item in parent_items(parent, groups):
        record = records[item["id"]]
        path = folder / "features" / (item["id"] + ".npz")
        if (
            record["parent_cache_sha256"] != item["cache_sha256"]
            or record["audio_sha256"] != item["audio_sha256"]
            or record["source_group"] != item["source_group"]
            or record["group"] != item["group"]
            or digest(path) != record["cache_sha256"]
        ):
            raise ValueError("Changed periodicity parent/cache identity")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["events"], item["events"])
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed periodicity plan identity")
            item["periodicity"] = saved["periodicity"]
        result.append(validate(item))
    prepared = json.loads((nsynth / "plan.json").read_text())
    new = json.loads((nsynth / "manifest.json").read_text())
    if (
        prepared["baseline_hashes"] != hashes()
        or prepared["code_sha256"] != nsynth_contracts()
        or new["plan_sha256"] != digest(nsynth / "plan.json")
        or len(new["items"]) != 84
        or len({i["id"] for i in new["items"]}) != 84
        or not new["reserved_instruments_untouched"]
        or not prepared["no_user_audio_or_scores"]
        or {g: sum(i["group"] == g for i in new["items"]) for g in ("train", "validation")}
        != {"train": 66, "validation": 18}
        or any(i["group"] not in ("train", "validation") for i in new["items"])
    ):
        raise ValueError("Changed or leaking NSynth fitting candidates")
    for record in new["items"]:
        if record["group"] not in groups:
            continue
        path = nsynth / record["id"] / "features.npz"
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed NSynth waveform/cache")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != new["plan_sha256"]:
                raise ValueError("Changed NSynth interval identity")
            item = {
                **record,
                **{k: saved[k] for k in ("events", "base_x", "x", "periodicity", "eligible")},
            }
        item.update(
            reference=np.asarray(record["reference"], float).reshape(-1, 3),
            pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3),
        )
        result.append(validate(item))
    partitions = {g: {i["source_group"] for i in result if i["group"] == g} for g in groups}
    if len({i["id"] for i in result}) != len(result) or (
        "train" in groups
        and "validation" in groups
        and partitions["train"] & partitions["validation"]
    ):
        raise ValueError("Fitting source-group leakage")
    return result
