"""Frozen warm-fusion winner only; reuse all 392 consumed V25 acoustic observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.bass_attack_declutter import BassAttackDeclutter
from app.services.bass_harmonic import features
from bass_temporal_data import annotate
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from release_neighbor_features import features as neighbor_features
from score_declutter_local import evaluate
from teacher_embedding_model import VERSION, model, probabilities


def frozen(run):
    from train_teacher_embedding_v32 import contracts

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
        or plan["required_consumed_regressions"] != 392
        or not plan["no_test_used_for_selection"]
    ):
        raise ValueError("No frozen safe V32 winner or changed fitting contract")
    winner = selection["winner"]
    path = run / winner["checkpoint"]
    report = run / winner["profile"] / "validation.json"
    if (
        digest(path) != winner["checkpoint_sha256"]
        or digest(report) != winner["validation_sha256"]
        or json.loads((run / winner["profile"] / "selection.json").read_text()) != winner
        or winner["threshold"] not in plan["thresholds"]
        or winner["guardian_threshold"] not in plan["guardian_thresholds"]
    ):
        raise ValueError("Changed frozen checkpoint or validation witness")
    validation = json.loads(report.read_text())
    for name in ("raw", "margin"):
        row = validation[name]
        if (
            not row["passes"]
            or row["false_notes_removed"] <= 0
            or len(row["per_recording"]) != 206
            or len({i["id"] for i in row["per_recording"]}) != 206
            or any(not i["passes"] for i in row["per_recording"])
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError("Failed or incomplete V32 validation preservation gates")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved["version"] != VERSION
        or saved["width"] != winner["width"]
        or saved["cap"] != winner["cap"]
    ):
        raise ValueError("Changed recurrence checkpoint contract")
    network = model(saved["width"], saved["cap"])
    network.load_state_dict(saved["state_dict"], strict=True)
    if digest(Path(plan["warm_start"])) != plan["warm_start_sha256"]:
        raise ValueError("Changed approved teacher source")
    approved = torch.load(Path(plan["warm_start"]), map_location="cpu", weights_only=True)
    if any(
        not torch.equal(v, approved["state_dict"][k])
        for k, v in network.teacher.state_dict().items()
    ):
        raise ValueError("Frozen teacher weights changed in selected checkpoint")
    for key in ("mean", "scale"):
        np.testing.assert_array_equal(saved[key].numpy()[:84], approved[key].numpy())
    return plan, winner, network, (saved["mean"].numpy(), saved["scale"].numpy())


def evaluate_run(run):
    from bass_joint_timing_v18 import margin

    plan, winner, network, normalizer = frozen(run)
    target = run / "consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed V32 regression; no retuning")
    from evaluate_periodicity_v26 import consumed_inputs as previous_inputs

    parent = previous_inputs()
    observations = Path(plan["periodicity_observation_root"])
    if (
        digest(observations / "manifest.json") != plan["periodicity_observation_manifest_sha256"]
        or digest(observations / "plan.json") != plan["periodicity_observation_plan_sha256"]
    ):
        raise ValueError("Changed consumed recurrence observations")
    observation_manifest = json.loads((observations / "manifest.json").read_text())
    records = {r["id"]: r for r in observation_manifest["items"]}
    if len(records) != 392:
        raise ValueError("Incomplete fusion regression observations")
    verifier = BassAttackDeclutter()
    items = []
    for item in parent:
        record = records.pop(item["id"])
        path = observations / (item["id"] + ".npz")
        if (
            digest(path) != record["cache_sha256"]
            or record["parent_cache_sha256"] != item["cache_sha256"]
        ):
            raise ValueError("Changed consumed recurrence cache/parent")
        p = verifier.model.probability(
            item["frames"], item["attack_frames"], item["release_frames"], item["x"]
        )
        keep = verifier.model.keep(
            item["events"], p, verifier.guardian.probability(item["base_x"]), item["duration"]
        )
        row = {
            **item,
            **{
                k: item[k][keep]
                for k in ("events", "base_x", "frames", "attack_frames", "release_frames")
            },
        }
        row.update(
            x=features(row["events"], row["base_x"]),
            eligible=eligible(row["events"], row["duration"]),
        )
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != observation_manifest["plan_sha256"]:
                raise ValueError("Changed regression observation clock identity")
            for key in ("events", "base_x", "x", "eligible"):
                np.testing.assert_array_equal(saved[key], row[key])
            row["periodicity"] = saved["periodicity"]
        row["release_neighbors"] = neighbor_features(row["events"], row["base_x"])
        items.append(annotate(row))
    if records:
        raise ValueError("Missing fusion regression record")
    ps = [probabilities(network, i, normalizer) for i in items]
    guards = [verifier.guardian.probability(i["base_x"]) for i in items]
    memo = {}
    raw = evaluate(items, ps, guards, winner["threshold"], winner["guardian_threshold"], memo)
    stricter = evaluate(
        items, ps, guards, margin(winner["threshold"]), winner["guardian_threshold"], memo
    )
    frozen(run)
    result = {
        "passes": raw["passes"] and stricter["passes"],
        "raw": raw,
        "margin": stricter,
        "recordings": 392,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "observation_manifest_sha256": digest(observations / "manifest.json"),
        "test_used_for_selection": False,
        "no_user_audio_or_scores": True,
        "new_weights_deployed": False,
    }
    preserve(target, result)
    print(json.dumps({k: result[k] for k in ("passes", "recordings")}, indent=2), flush=True)


def require_regression(run):
    _, winner, _, _ = frozen(run)
    report = json.loads((run / "consumed-regression.json").read_text())
    if (
        not report["passes"]
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or report["batch_selection_sha256"] != digest(run / "batch-selection.json")
        or report["recordings"] != 392
    ):
        raise ValueError("V32 regression failed; do not generate fresh audio or export weights")
    for name in ("raw", "margin"):
        if len(report[name]["per_recording"]) != 392 or any(
            not i["passes"] for i in report[name]["per_recording"]
        ):
            raise ValueError("Incomplete V32 per-recording regression gates")
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evaluate_run(args.run)
