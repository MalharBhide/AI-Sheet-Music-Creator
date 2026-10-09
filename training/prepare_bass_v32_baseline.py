"""Replay the shipped V32 rejection on verified fitting caches without audio inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_embedding_declutter import BassEmbeddingDeclutter
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v32_baseline import ASSET_SHA, VERSION, hashes
from prepare_robust_training_stems import digest, preserve
from release_neighbor_features import features as neighbors

COUNTS = {"train": 702, "validation": 206}


def prepare(baseline, mixtures, output):
    # The declared V25 replay imports the original guarded fitting readers.
    # Its current_bass_v25_baseline checks the untouched snapshot, not V32 files.
    from groove_bass_data import load

    current = hashes()
    items = load(baseline, mixtures)
    if len(items) != 908 or {g: sum(i["group"] == g for i in items) for g in COUNTS} != COUNTS:
        raise ValueError("Incomplete V32 fitting source coverage")
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    plan = {
        "version": VERSION,
        "baseline_hashes": current,
        "baseline_root": str(baseline),
        "mixtures_root": str(mixtures),
        "baseline_manifest_sha256": digest(baseline / "manifest.json"),
        "mixtures_manifest_sha256": digest(mixtures / "manifest.json"),
        "producer_sha256": digest(Path(__file__)),
        "groups": COUNTS,
        "legacy_code_replay": "v32-frozen-training-code with original V25 route and producer guards",
        "no_audio_inference": True,
        "no_regression_read": True,
        "no_user_audio_or_scores": True,
        "evidence_policy": "Retained acoustic/recurrence windows and note clocks are unchanged. Recompute event-neighbor features and context only on actual V32 survivors.",
    }
    preserve(output / "plan.json", plan)
    folder = output / "features"
    folder.mkdir()
    verifier = BassEmbeddingDeclutter(root / "backend/app/assets/bass-embedding-v1.npz", ASSET_SHA)
    records = []
    for item in items:
        p = verifier.model.probability(item)
        keep = verifier.model.keep(
            item["events"], p, verifier.guardian.probability(item["base_x"]), item["duration"]
        )
        events, base = item["events"][keep], item["base_x"][keep]
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
            v32_removed=int((~keep).sum()),
            notes=len(events),
        )
        path = folder / (item["id"] + ".npz")
        arrays = {
            k: item[k][keep] for k in ("frames", "attack_frames", "release_frames", "periodicity")
        }
        np.savez_compressed(
            path,
            **arrays,
            events=events,
            base_x=base,
            x=features(events, base),
            release_neighbors=neighbors(events, base),
            eligible=eligible(events, item["duration"]),
            identity=identity(record),
            plan_sha256=digest(output / "plan.json"),
        )
        record["cache_sha256"] = digest(path)
        records.append(record)
    if current != hashes():
        raise ValueError("V32 changed during baseline preparation")
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
                "removed": sum(i["v32_removed"] for i in records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("mixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.baseline, args.mixtures, args.output)
