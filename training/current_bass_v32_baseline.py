"""Explicit released V32 contract; historical V25 guards remain unchanged."""

import json
from pathlib import Path

from prepare_robust_training_stems import digest

VERSION = "shipped-bass-v32-baseline-v1"
RELEASE_SHA = "728a9a6a2748bd18a2d57c2101f4cae0d8791c901e9b8875800be0192c572775"
PLAN_SHA = "030d63d998868dad26e909f17922acd701d2c328a87441b1828e69e60615e9fd"
ASSET_SHA = "edc91517f39eef69e85a4c945b8fd0416d6973f952a9dcf4f8d139dc364a0955"


def hashes():
    root = Path(__file__).resolve().parents[1]
    folder = root / "training/results/teacher-embedding-v32-v1"
    if (
        digest(folder / "release-checks.json") != RELEASE_SHA
        or digest(folder / "plan.json") != PLAN_SHA
    ):
        raise ValueError("Changed released V32 evidence; declare a new baseline")
    release = json.loads((folder / "release-checks.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    if (
        release["deployment"]["status"] != "verified"
        or release["full_backend_training_tests"]["passed"] != 832
        or release["native_integration"]["passed"] != 6
        or not release["native_integration"]["staged_module_origins_asserted"]
        or not release["new_weights_deployed"]
        or not release["no_user_audio_or_scores"]
        or release["deployment"]["running_model_check"]["asset_sha256"] != ASSET_SHA
    ):
        raise ValueError("Incomplete V32 release witness")
    inherited = plan["baseline_hashes"]
    changed = {"backend/app/services/piano_transcription.py", "backend/app/routes/dependencies.py"}
    sources = {}
    for name, expected in inherited["sources"].items():
        if name not in changed:
            if digest(root / name) != expected:
                raise ValueError("Changed inherited V25 implementation or array")
            sources[name] = expected
    for name, expected in release["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Changed parity-checked V32 implementation or array")
        sources[name] = expected
    for name, expected in inherited["arrays"].items():
        if digest(root / "backend/app/assets" / name) != expected:
            raise ValueError("Changed inherited V25 model array")
    return {
        "version": VERSION,
        "sources": sources,
        "arrays": {**inherited["arrays"], "bass-embedding-v1.npz": ASSET_SHA},
        "release_sha256": RELEASE_SHA,
        "thresholds": {"remove": 0.8, "guardian": 0.5},
        "preceding_source_commit": "d214a3e",
        "all_eight_v25_stages_retained": True,
    }
