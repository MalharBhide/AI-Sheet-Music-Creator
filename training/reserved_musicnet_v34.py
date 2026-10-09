"""Freeze reserved work identities without reading their audio or note labels."""

import json
from pathlib import Path

from prepare_musicnet_v32 import sources
from prepare_robust_training_stems import digest


def declaration(piano):
    observation = json.loads((piano / "plan.json").read_text())
    source = Path(observation["source_root"])
    if digest(source / "manifest.json") != observation["source_manifest_sha256"]:
        raise ValueError("Changed MusicNet source manifest")
    rows = sources(source)
    reserved = [row for row in rows if row["group"] == "reserved"]
    if not reserved:
        raise ValueError("No unused reserved MusicNet works")
    jobs = [
        {
            "id": f"musicnet-v34-reserved-{row['id']}-{index}",
            "publisher_id": row["id"],
            "work": row["work"],
            "labels": row["labels"],
            "labels_sha256": row["labels_sha256"],
            "clock": row["clock"],
            "excerpt_index": index,
            "excerpt": crop,
        }
        for row in reserved
        for index, crop in enumerate(row["excerpts"])
    ]
    return {
        "source_root": str(source),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "jobs": jobs,
        "procedural_seeds": list(range(279101, 279133)),
        "expected_recordings": len(jobs) + 32,
        "no_reserved_audio_or_labels_read": True,
    }
