"""Apply shipped V25 to sealed earlier caches; no decoder inference or user audio."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_attack_declutter import BassAttackDeclutter
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v25_baseline import VERSION, hashes
from prepare_robust_training_stems import digest, preserve

COUNTS = {"train": 573, "validation": 160}


def prepare(parent, output):
    # Run with the read-only e56554c replay code first on PYTHONPATH. Its existing
    # loaders check every legacy producer/source hash without relaxing guards.
    from release_local_bass_data import load

    current = hashes()
    items = load(parent)
    if len(items) != 733 or {g: sum(i["group"] == g for i in items) for g in COUNTS} != COUNTS:
        raise ValueError("Incomplete V25 fitting source coverage")
    contract = {
        "version": VERSION,
        "baseline_hashes": current,
        "parent_root": str(parent),
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "parent_plan_sha256": digest(parent / "plan.json"),
        "producer_sha256": digest(Path(__file__)),
        "groups": COUNTS,
        "legacy_code_replay": "e56554c with archived parity-executed V25 runtime",
        "no_audio_inference": True,
        "no_regression_read": True,
        "no_user_audio_or_scores": True,
    }
    output.mkdir(exist_ok=False)
    preserve(output / "plan.json", contract)
    folder = output / "features"
    folder.mkdir()
    verifier = BassAttackDeclutter()
    records = []
    for item in items:
        probability = verifier.model.probability(
            item["frames"], item["attack_frames"], item["release_frames"], item["x"]
        )
        guardian = verifier.guardian.probability(item["base_x"])
        keep = verifier.model.keep(item["events"], probability, guardian, item["duration"])
        events, base_x = item["events"][keep], item["base_x"][keep]
        record = {
            k: item[k]
            for k in (
                "id",
                "group",
                "source_group",
                "corpus",
                "audio",
                "audio_sha256",
                "duration",
                "seconds",
            )
        }
        record.update(
            reference=item["reference"].tolist(),
            pitch_reference=item["pitch_reference"].tolist(),
            parent_cache_sha256=item["cache_sha256"],
            v25_removed=int((~keep).sum()),
            notes=len(events),
        )
        path = folder / (item["id"] + ".npz")
        np.savez_compressed(
            path,
            events=events,
            base_x=base_x,
            x=features(events, base_x),
            eligible=eligible(events, item["duration"]),
            identity=identity(record),
            plan_sha256=digest(output / "plan.json"),
        )
        record["cache_sha256"] = digest(path)
        records.append(record)
    if current != hashes():
        raise ValueError("V25 changed during baseline preparation")
    preserve(
        output / "manifest.json",
        {
            "version": VERSION,
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "no_audio_inference": True,
            "no_regression_read": True,
            "no_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "recordings": len(records),
                "removed": sum(i["v25_removed"] for i in records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.output)
