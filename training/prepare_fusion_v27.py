"""Combine frozen V25 ending/attack observations with new waveform recurrence."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_attack_declutter import BassAttackDeclutter
from app.services.temporal_note_model import sequences
from attack_local_features import observe
from attack_local_features import sequences as attacks
from current_bass_v25_baseline import hashes
from periodicity_bass_data import load
from prepare_robust_training_stems import digest, preserve
from release_local_features import release_sequences

VERSION = "attack-ending-periodicity-fusion-v27-v1"


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/prepare_fusion_v27.py",
        "training/attack_local_features.py",
        "training/release_local_features.py",
        "training/periodicity_bass_data.py",
    )
    return {name: digest(root / name) for name in names}


def prepare(periodicity, nsynth, output):
    from release_local_bass_data import load as legacy_items

    items = load(periodicity, nsynth)
    # Replay with the unchanged e56554c code to keep every old raw-route guard.
    legacy = {i["id"]: i for i in legacy_items(periodicity.parent / "release-local-bass-v1")}
    if len(items) != 817 or len(legacy) != 733:
        raise ValueError("Incomplete fusion fitting source inventory")
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "periodicity_root": str(periodicity),
        "periodicity_manifest_sha256": digest(periodicity / "manifest.json"),
        "nsynth_root": str(nsynth),
        "nsynth_manifest_sha256": digest(nsynth / "manifest.json"),
        "legacy_root": str(periodicity.parent / "release-local-bass-v1"),
        "legacy_manifest_sha256": digest(
            periodicity.parent / "release-local-bass-v1/manifest.json"
        ),
        "groups": {"train": 639, "validation": 178},
        "no_regression_audio_or_metrics": True,
        "no_user_audio_or_transcription_inference": True,
    }
    output.mkdir(exist_ok=False)
    preserve(output / "plan.json", plan)
    folder = output / "features"
    folder.mkdir()
    verifier = BassAttackDeclutter()
    records = []
    for item in items:
        if item["id"] in legacy:
            old = legacy.pop(item["id"])
            p = verifier.model.probability(
                old["frames"], old["attack_frames"], old["release_frames"], old["x"]
            )
            keep = verifier.model.keep(
                old["events"], p, verifier.guardian.probability(old["base_x"]), old["duration"]
            )
            np.testing.assert_array_equal(item["events"], old["events"][keep])
            values = {k: old[k][keep] for k in ("frames", "attack_frames", "release_frames")}
        else:
            if not item["source_group"].startswith("nsynth-"):
                raise ValueError("Unexpected new fusion source")
            wave, rate = sf.read(item["audio"], dtype="float32")
            if (
                rate != 22050
                or len(wave) / rate != item["duration"]
                or digest(Path(item["audio"])) != item["audio_sha256"]
            ):
                raise ValueError("Changed new instrument physical clock")
            observed = observe(wave, rate)
            values = {
                "frames": sequences(observed["cqt"], item["events"]),
                "attack_frames": attacks(observed, item["events"]),
                "release_frames": release_sequences(observed, item["events"]),
            }
        path = folder / (item["id"] + ".npz")
        np.savez_compressed(
            path, **values, events=item["events"], plan_sha256=digest(output / "plan.json")
        )
        records.append(
            {
                "id": item["id"],
                "source_group": item["source_group"],
                "group": item["group"],
                "parent_cache_sha256": item["cache_sha256"],
                "cache_sha256": digest(path),
            }
        )
        if len(records) % 20 == 0:
            print(json.dumps({"cached": len(records), "total": 817}), flush=True)
    if legacy or plan["code_sha256"] != contracts() or plan["baseline_hashes"] != hashes():
        raise ValueError("Fusion observations changed or lost fitting records")
    preserve(
        output / "manifest.json",
        {
            "version": VERSION,
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "no_regression_audio_or_metrics": True,
            "no_user_audio_or_transcription_inference": True,
        },
    )
    print(
        json.dumps(
            {"passes": True, "recordings": 817, "manifest_sha256": digest(output / "manifest.json")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("periodicity", type=Path)
    parser.add_argument("nsynth", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.periodicity, args.nsynth, args.output)
