"""Read only pinned fitting mixtures and observed long-hold release neighbors."""

import json
from pathlib import Path

import numpy as np
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from fusion_bass_data import load as fitting_items
from periodicity_bass_data import validate
from prepare_groove_bass import contracts, drum_sources
from prepare_nsynth_bass import sources
from prepare_robust_training_stems import digest
from release_neighbor_features import features


def load(baseline, mixtures, groups=("train", "validation")):
    if not groups or any(g not in ("train", "validation") for g in groups):
        raise ValueError("Groove fitting cannot read reserved or consumed test recordings")
    plan = json.loads((mixtures / "plan.json").read_text())
    manifest = json.loads((mixtures / "manifest.json").read_text())
    if (
        manifest["plan_sha256"] != digest(mixtures / "plan.json")
        or plan["baseline_hashes"] != hashes()
        or plan["code_sha256"] != contracts()
        or not plan["reserved_drum_performers_untouched"]
        or not manifest["reserved_drum_performers_untouched"]
        or not manifest["reserved_instruments_untouched"]
        or not plan["drum_midi_is_not_piano_pitch_truth"]
        or not plan["no_user_audio_or_scores"]
        or not manifest["no_user_audio_or_scores"]
        or len(manifest["items"]) != 91
        or len({r["id"] for r in manifest["items"]}) != 91
        or {g: sum(r["group"] == g for r in manifest["items"]) for g in ("train", "validation")}
        != {"train": 63, "validation": 28}
        or any(r["group"] not in ("train", "validation") for r in manifest["items"])
    ):
        raise ValueError("Changed, incomplete or leaking Groove fitting mixtures")
    drum, nsynth = Path(plan["source_root"]), Path(plan["nsynth_root"])
    if (
        digest(drum / "manifest.json") != plan["source_manifest_sha256"]
        or digest(nsynth / "manifest.json") != plan["nsynth_manifest_sha256"]
    ):
        raise ValueError("Changed licensed mixture source manifests")
    drummers = {r["id"]: r for r in drum_sources(drum)}
    instruments = {r["id"]: r for r in sources(nsynth)}
    result = [i for i in fitting_items(baseline) if i["group"] in groups]
    for item in result:
        item["release_neighbors"] = features(item["events"], item["base_x"])
    for record in manifest["items"]:
        if record["group"] not in groups:
            continue
        source = drummers[record["drum_source"]]
        if (
            source["group"] != record["group"]
            or source["drummer"] != record["drummer"]
            or source["audio_sha256"] != record["drum_audio_sha256"]
            or any(
                instruments[name]["group"] != record["group"]
                or instruments[name]["instrument_str"] != record["instrument_str"]
                for name in record["source_notes"]
            )
        ):
            raise ValueError("Cross-partition performer or pitched source leakage")
        if (
            record["source_group"]
            != "groove-" + source["drummer"] + "/nsynth-" + record["instrument_str"]
        ):
            raise ValueError("Changed joint source-group identity")
        cache = mixtures / record["id"] / "features.npz"
        if (
            digest(cache) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
            or record["plan_sha256"] != manifest["plan_sha256"]
        ):
            raise ValueError("Changed physical mixture cache or waveform")
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed mixture observation clock identity")
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
                raise ValueError("Invalid physical mixture encoder windows")
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
        if item["kind"] == "drums_only" and (
            len(item["reference"]) or len(item["pitch_reference"])
        ):
            raise ValueError("Drum instrument numbers must not become piano pitch truth")
        result.append(validate(item))
    return result
