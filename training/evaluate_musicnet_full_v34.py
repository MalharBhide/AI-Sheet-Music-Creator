"""Evaluate the frozen real-piano winner on all 469 released-V32 survivor caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.bass_attack_declutter import BassAttackDeclutter
from current_bass_v32_baseline import hashes
from musicnet_full_encoding_model import (
    VERSION,
    WARM_SHA,
    model,
    normalizer,
    probabilities,
)
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import evaluate


def frozen(run):
    from bass_joint_timing_v18 import margin
    from reserved_musicnet_v34 import declaration
    from train_musicnet_full_v34 import contracts

    plan = json.loads((run / "plan.json").read_text())
    selection = json.loads((run / "batch-selection.json").read_text())
    if (
        plan["version"] != VERSION
        or plan["baseline_hashes"] != hashes()
        or plan["code_sha256"] != contracts()
        or selection["plan_sha256"] != digest(run / "plan.json")
        or not selection["selected"]
        or selection["test_used_for_selection"]
        or not selection["no_user_audio_or_scores"]
        or plan["required_consumed_regressions"] != 469
        or not plan["no_test_used_for_selection"]
        or plan["warm_start_sha256"] != WARM_SHA
    ):
        raise ValueError("No frozen safe V34 winner or changed fitting contract")
    if (
        digest(Path(plan["baseline_root"]) / "manifest.json")
        != plan["baseline_manifest_sha256"]
        or digest(Path(plan["piano_root"]) / "manifest.json")
        != plan["piano_manifest_sha256"]
        or digest(Path(plan["piano_root"]) / "plan.json") != plan["piano_plan_sha256"]
        or declaration(Path(plan["piano_root"])) != plan["first_pass_declaration"]
    ):
        raise ValueError("Changed frozen fitting or reserved declaration")
    winner = selection["winner"]
    path = run / winner["checkpoint"]
    report = run / winner["profile"] / "validation.json"
    if (
        digest(path) != winner["checkpoint_sha256"]
        or digest(report) != winner["validation_sha256"]
        or json.loads((run / winner["profile"] / "selection.json").read_text())
        != winner
        or winner["threshold"] not in plan["thresholds"]
        or winner["guardian_threshold"] not in plan["guardian_thresholds"]
    ):
        raise ValueError("Changed frozen checkpoint or validation witness")
    validation = json.loads(report.read_text())
    for name, threshold in (
        ("raw", winner["threshold"]),
        ("margin", margin(winner["threshold"])),
    ):
        row = validation[name]
        if (
            not row["passes"]
            or row["remove_threshold"] != threshold
            or row["guardian_threshold"] != winner["guardian_threshold"]
            or row["false_notes_removed"] <= 0
            or len(row["per_recording"])
            != plan["expected_fitting_counts"]["validation"]
            or len({i["id"] for i in row["per_recording"]})
            != plan["expected_fitting_counts"]["validation"]
            or any(not i["passes"] for i in row["per_recording"])
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError("Failed or incomplete V34 validation preservation gates")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved["version"] != VERSION
        or saved["width"] != winner["width"]
        or saved["cap"] != winner["cap"]
        or saved["epoch"] != winner["epoch"]
    ):
        raise ValueError("Changed recurrence checkpoint contract")
    network = model(saved["width"], saved["cap"])
    network.load_state_dict(saved["state_dict"], strict=True)
    if digest(Path(plan["warm_start"])) != plan["warm_start_sha256"]:
        raise ValueError("Changed approved teacher source")
    approved = torch.load(
        Path(plan["warm_start"]), map_location="cpu", weights_only=True
    )
    if any(
        not torch.equal(v, approved["state_dict"][k])
        for k, v in network.anchor.state_dict().items()
    ):
        raise ValueError("Frozen teacher weights changed in selected checkpoint")
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(saved[key].numpy(), approved[key].numpy())
    return plan, winner, network, normalizer(saved)


def evaluate_run(run):
    from bass_joint_timing_v18 import margin

    plan, winner, network, normalizer = frozen(run)
    target = run / "consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed V34 regression; no retuning")
    from bass_v32_regression_data import load

    observations = Path(plan["regression_root"])
    if digest(observations / "plan.json") != plan["regression_plan_sha256"]:
        raise ValueError("Changed consumed V32 preparation plan")
    items = load(observations, plan["regression_manifest_sha256"])
    verifier = BassAttackDeclutter()
    ps = [probabilities(network, i, normalizer) for i in items]
    guards = [verifier.guardian.probability(i["base_x"]) for i in items]
    memo = {}
    raw = evaluate(
        items, ps, guards, winner["threshold"], winner["guardian_threshold"], memo
    )
    stricter = evaluate(
        items,
        ps,
        guards,
        margin(winner["threshold"]),
        winner["guardian_threshold"],
        memo,
    )
    frozen(run)
    result = {
        "passes": raw["passes"] and stricter["passes"],
        "raw": raw,
        "margin": stricter,
        "recordings": 469,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "regression_manifest_sha256": digest(observations / "manifest.json"),
        "test_used_for_selection": False,
        "no_user_audio_or_scores": True,
        "new_weights_deployed": False,
    }
    preserve(target, result)
    print(
        json.dumps({k: result[k] for k in ("passes", "recordings")}, indent=2),
        flush=True,
    )


def require_regression(run):
    _, winner, _, _ = frozen(run)
    report = json.loads((run / "consumed-regression.json").read_text())
    if (
        not report["passes"]
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or report["batch_selection_sha256"] != digest(run / "batch-selection.json")
        or report["recordings"] != 469
    ):
        raise ValueError(
            "V34 regression failed; do not generate fresh audio or export weights"
        )
    for name in ("raw", "margin"):
        if len(report[name]["per_recording"]) != 469 or any(
            not i["passes"] for i in report[name]["per_recording"]
        ):
            raise ValueError("Incomplete V34 per-recording regression gates")
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evaluate_run(args.run)
