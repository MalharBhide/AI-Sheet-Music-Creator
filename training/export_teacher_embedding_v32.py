"""Export only the frozen winner after complete regression and first-pass success."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_teacher_embedding_v32 import frozen, require_regression
from prepare_robust_training_stems import digest, preserve
from teacher_embedding_model import VERSION


def export(run, fixtures, output):
    plan, winner, network, normalizer = frozen(run)
    require_regression(run)
    report = json.loads((run / "original-first-pass.json").read_text())
    manifest = json.loads((fixtures / "manifest.json").read_text())
    fixture_plan = json.loads((fixtures / "plan.json").read_text())
    ids = [r["id"] for r in manifest["items"]]
    root = Path(__file__).resolve().parents[1]
    if (
        not report["passes"]
        or not report["first_pass_complete"]
        or report["recordings"] != 77
        or len(ids) != 77
        or len(set(ids)) != 77
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or report["batch_selection_sha256"] != digest(run / "batch-selection.json")
        or report["test_manifest_sha256"] != digest(fixtures / "manifest.json")
        or manifest["plan_sha256"] != digest(fixtures / "plan.json")
        or fixture_plan["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or fixture_plan["consumed_regression_sha256"] != digest(run / "consumed-regression.json")
        or fixture_plan["procedural_seeds"] != plan["new_first_pass_seeds"]
        or any(digest(root / name) != sha for name, sha in fixture_plan["code_sha256"].items())
    ):
        raise ValueError("Failed or changed independent first-pass release evidence")
    for name in ("raw", "margin"):
        row = report[name]
        if (
            not row["passes"]
            or row["false_notes_removed"] <= 0
            or [r["id"] for r in row["per_recording"]] != ids
            or any(not r["passes"] for r in row["per_recording"])
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError("Incomplete independent preservation/noise-reduction gates")
    for record in manifest["items"]:
        if (
            digest(fixtures / record["id"] / "features.npz") != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed independent physical observations")
    if output.exists():
        raise ValueError("Preserve existing exported asset")
    values = {
        "version": np.asarray(VERSION),
        "format_version": np.asarray(3),
        "width": np.asarray(winner["width"]),
        "cap": np.asarray(winner["cap"]),
        "remove_threshold": np.asarray(winner["threshold"]),
        "guardian_threshold": np.asarray(winner["guardian_threshold"]),
        "mean": normalizer[0].astype(np.float32),
        "scale": normalizer[1].astype(np.float32),
    }
    values.update(
        {"weight_" + k: v.numpy().astype(np.float32) for k, v in network.state_dict().items()}
    )
    np.savez_compressed(output, **values)
    preserve(
        run / "portable-export.json",
        {
            "asset_sha256": digest(output),
            "asset": str(output),
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "first_pass_sha256": digest(run / "original-first-pass.json"),
            "no_user_audio_or_scores": True,
            "new_weights_deployed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.run, args.fixtures, args.output)
