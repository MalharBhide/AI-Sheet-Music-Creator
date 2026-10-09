"""Join verified real-piano observations with licensed fitting caches; forbid reserve."""

import json
from pathlib import Path

import numpy as np
from bass_training_data import eligible
from bass_v32_data import load as previous_items
from current_bass_v32_baseline import hashes
from periodicity_bass_data import validate
from prepare_musicnet_v32 import contracts, sources
from prepare_robust_training_stems import digest
from release_neighbor_features import features


def load(baseline, piano, groups=("train", "validation")):
    if not groups or any(g not in ("train", "validation") for g in groups):
        raise ValueError("MusicNet fitting cannot read reserved or consumed test recordings")
    plan = json.loads((piano / "plan.json").read_text())
    manifest = json.loads((piano / "manifest.json").read_text())
    if (
        manifest["plan_sha256"] != digest(piano / "plan.json")
        or plan["code_sha256"] != contracts()
        or plan["baseline_hashes"] != hashes()
        or not plan["reserved_works_untouched"]
        or not manifest["reserved_works_untouched"]
        or not manifest["no_user_audio_or_scores"]
        or not plan["no_user_audio_or_scores"]
    ):
        raise ValueError("Changed or incomplete MusicNet fitting observations")
    source = Path(plan["source_root"])
    if digest(source / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("Changed original MusicNet source manifest")
    originals = {r["id"]: r for r in sources(source)}
    records = manifest["items"]
    expected = {
        g: sum(len(r["excerpts"]) for r in originals.values() if r["group"] == g)
        for g in ("train", "validation")
    }
    if (
        expected != plan["expected_counts"]
        or len({r["id"] for r in records}) != len(records)
        or expected != {g: sum(r["group"] == g for r in records) for g in expected}
        or any(r["group"] not in expected for r in records)
    ):
        raise ValueError("Incomplete or leaking real-piano fitting partition")
    result = previous_items(baseline, groups)
    for record in records:
        if record["group"] not in groups:
            continue
        original = originals[record["publisher_id"]]
        crop = original["excerpts"][record["source_excerpt_index"]]
        if (
            original["group"] != record["group"]
            or record["source_group"] != "musicnet/" + "/".join(original["work"])
            or record["labels_sha256"] != original["labels_sha256"]
            or record["source_audio_sha256"] != crop["audio_sha256"]
        ):
            raise ValueError("MusicNet source clock, note provenance or partition mismatch")
        path = piano / record["id"] / "features.npz"
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
            or record["plan_sha256"] != manifest["plan_sha256"]
        ):
            raise ValueError("Changed physical MusicNet observation")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed real-piano observation identity")
            item = {
                **record,
                **{
                    k: saved[k]
                    for k in (
                        "events",
                        "base_x",
                        "x",
                        "periodicity",
                        "eligible",
                        "frames",
                        "attack_frames",
                        "release_frames",
                        "release_neighbors",
                    )
                },
            }
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
                raise ValueError("Invalid real-piano acoustic windows")
        np.testing.assert_array_equal(
            item["eligible"], eligible(item["events"], record["duration"])
        )
        np.testing.assert_array_equal(
            item["release_neighbors"], features(item["events"], item["base_x"])
        )
        item.update(
            reference=np.asarray(record["reference"], float).reshape(-1, 3),
            pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3),
        )
        result.append(validate(item))
    partitions = {g: {r["source_group"] for r in result if r["group"] == g} for g in groups}
    if len({r["id"] for r in result}) != len(result) or (
        "train" in groups
        and "validation" in groups
        and partitions["train"] & partitions["validation"]
    ):
        raise ValueError("Combined fitting source leakage")
    return result
