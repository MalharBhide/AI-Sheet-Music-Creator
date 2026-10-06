"""Explicit V25 contract; old V16 source guards are deliberately not weakened."""

import json
from pathlib import Path

from prepare_robust_training_stems import digest

VERSION = "shipped-bass-v25-baseline-v1"
ROUTE_SHA = "ca07ef6e53e2ffb5a2609503a3c0f23ef94d79f9d533616965f1b6cb119c93dc"
RELEASE_SHA = "a91c801f49f3b311d0a039692b6e66c45191074d630b66f3467450df5399c31a"


def hashes():
    root = Path(__file__).resolve().parents[1]
    release = root / "training/results/attack-release-v25-v1/release-checks.json"
    if (
        digest(release) != RELEASE_SHA
        or digest(root / "backend/app/services/piano_transcription.py") != ROUTE_SHA
    ):
        raise ValueError("Changed deployed V25 release; declare a new baseline")
    saved = json.loads(release.read_text())
    if (
        saved["deployment"]["status"] != "verified"
        or saved["full_backend_training_tests"]["passed"] != 742
    ):
        raise ValueError("V25 deployment witness incomplete")
    for name, expected in saved["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Changed deployed V25 source or array")
    proof = json.loads(
        (
            root / "training/results/attack-release-v25-v1/runtime-installation-source-proof.json"
        ).read_text()
    )
    sources = dict(saved["source_sha256"])
    for name, entry in proof.items():
        path = "backend/app/services/" + name + ".py"
        if digest(root / path) != entry["installed_source_sha256"]:
            raise ValueError("Changed parity-checked V25 runtime")
        sources[path] = entry["installed_source_sha256"]
    # Pin the complete existing bass runtime, not only the new stage.
    witness = json.loads((root / "training/results/attack-release-v25-v1/plan.json").read_text())
    for name, expected in witness["baseline_hashes"]["arrays"].items():
        if digest(root / "backend/app/assets" / name) != expected:
            raise ValueError("Changed inherited bass array")
    for name, expected in witness["baseline_hashes"]["sources"].items():
        if name == "backend/app/services/piano_transcription.py":
            continue
        if digest(root / name) != expected:
            raise ValueError("Changed inherited V16 implementation")
        sources[name] = expected
    return {
        "version": VERSION,
        "sources": sources,
        "release_sha256": RELEASE_SHA,
        "arrays": {
            **witness["baseline_hashes"]["arrays"],
            "bass-attack-declutter-v1.npz": saved["source_sha256"][
                "backend/app/assets/bass-attack-declutter-v1.npz"
            ],
        },
        "thresholds": {"remove": 0.8, "guardian": 0.5},
        "preceding_source_commit": "e56554c",
    }
