"""Restore the pinned historical reader in an isolated tree; never weaken guards."""

import argparse
import json
import shutil
from pathlib import Path

from prepare_robust_training_stems import digest, preserve


def prepare(output):
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "training/results/release-only-v38-v1/plan.json"
    plan = json.loads(plan_path.read_text())
    original = root / ".training/release-only-v38-original-runtime"
    if not original.is_dir():
        # /src is mounted read-only while its .training mount appears as /training.
        original = Path("/training/release-only-v38-original-runtime")
    if output.exists():
        raise ValueError("Preserve existing frozen replay")
    shutil.copytree(root / "backend", output / "backend",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", ".venv"))
    shutil.copytree(root / "training", output / "training",
                    ignore=shutil.ignore_patterns("results", "__pycache__", ".pytest_cache", ".ruff_cache"))
    (output / "training/results").symlink_to(root / "training/results", target_is_directory=True)
    for name, expected in plan["baseline_hashes"]["sources"].items():
        source = original / name
        if digest(source) != expected:
            raise ValueError("Historical V32 source differs from released V38 input contract")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    preserve(output / "replay.json", {"v38_plan_sha256": digest(plan_path),
        "original_v32_sources": plan["baseline_hashes"]["sources"],
        "historical_observations_only": True, "no_guard_changes": True,
        "no_user_audio_or_scores": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
