"""Export only the frozen winner after complete regression and first-pass success."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_musicnet_full_v34 import frozen, require_regression
from fresh_musicnet_full_v34 import regression_gate
from musicnet_full_encoding_model import VERSION
from prepare_robust_training_stems import digest, preserve


def export(run, fixtures, output):
    plan, winner, network, normalizer = frozen(run)
    require_regression(run)
    regression_gate(run, winner)
    from gate_musicnet_real_piano_v34 import require_real_piano

    real_piano = require_real_piano(run)
    report = json.loads((run / "original-first-pass.json").read_text())
    manifest = json.loads((fixtures / "manifest.json").read_text())
    fixture_plan = json.loads((fixtures / "plan.json").read_text())
    ids = [r["id"] for r in manifest["items"]]
    declared = plan["first_pass_declaration"]
    expected_ids = [j["id"] for j in declared["jobs"]] + [
        "musicnet-v34-fresh-" + str(seed) for seed in declared["procedural_seeds"]
    ]
    expected = declared["expected_recordings"]
    root = Path(__file__).resolve().parents[1]
    if (
        not report["passes"]
        or not report["first_pass_complete"]
        or report["recordings"] != expected
        or len(ids) != expected
        or len(set(ids)) != expected
        or ids != expected_ids
        or fixture_plan["first_pass_declaration"] != declared
        or not manifest["now_consumed_regression"]
        or not report["no_retuning"]
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or report["batch_selection_sha256"] != digest(run / "batch-selection.json")
        or report["test_manifest_sha256"] != digest(fixtures / "manifest.json")
        or manifest["plan_sha256"] != digest(fixtures / "plan.json")
        or fixture_plan["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or fixture_plan["consumed_regression_sha256"]
        != digest(run / "consumed-regression.json")
        or fixture_plan["procedural_seeds"] != plan["new_first_pass_seeds"]
        or any(
            digest(root / name) != sha
            for name, sha in fixture_plan["code_sha256"].items()
        )
    ):
        raise ValueError("Failed or changed independent first-pass release evidence")
    from bass_joint_timing_v18 import margin

    for name, threshold in (
        ("raw", winner["threshold"]),
        ("margin", margin(winner["threshold"])),
    ):
        row = report[name]
        if (
            not row["passes"]
            or row["remove_threshold"] != threshold
            or row["guardian_threshold"] != winner["guardian_threshold"]
            or row["false_notes_removed"] <= 0
            or [r["id"] for r in row["per_recording"]] != ids
            or any(not r["passes"] for r in row["per_recording"])
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError(
                "Incomplete independent preservation/noise-reduction gates"
            )
    for record in manifest["items"]:
        if (
            digest(fixtures / record["id"] / "features.npz") != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed independent physical observations")
    if output.suffix != ".npz":
        raise ValueError("Portable asset path must end in .npz")
    if output.exists():
        raise ValueError("Preserve existing exported asset")
    values = {
        "version": np.asarray(VERSION),
        "format_version": np.asarray(5),
        "width": np.asarray(winner["width"]),
        "cap": np.asarray(winner["cap"]),
        "remove_threshold": np.asarray(winner["threshold"]),
        "guardian_threshold": np.asarray(winner["guardian_threshold"]),
        "mean": normalizer[0].astype(np.float32),
        "scale": normalizer[1].astype(np.float32),
    }
    values.update(
        {
            "weight_" + k: v.numpy().astype(np.float32)
            for k, v in network.state_dict().items()
        }
    )
    np.savez_compressed(output, **values)
    preserve(
        run / "portable-export.json",
        {
            "asset_sha256": digest(output),
            "asset": str(output),
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "first_pass_sha256": digest(run / "original-first-pass.json"),
            "real_piano_release_gate": real_piano,
            "exporter_sha256": digest(Path(__file__)),
            "runtime_source_sha256": {
                name: digest(root / name)
                for name in (
                    "backend/app/services/bass_full_encoding_network.py",
                    "backend/app/services/bass_full_encoding_declutter.py",
                )
            },
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
