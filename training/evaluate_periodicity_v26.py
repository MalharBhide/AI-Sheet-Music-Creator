"""Frozen validation winner only; replay all 392 consumed V25 baseline recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.services.bass_attack_declutter import BassAttackDeclutter
from app.services.bass_harmonic import features
from bass_temporal_data import annotate
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from periodicity_features import features as recurrence_features
from periodicity_features import observe
from periodicity_model import VERSION, inputs, model, probabilities
from prepare_robust_training_stems import digest, preserve
from score_declutter_local import evaluate


def frozen(run):
    from train_periodicity_v26 import contracts

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
        raise ValueError("No frozen safe V26 winner or changed fitting contract")
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
            or len(row["per_recording"]) != 178
            or len({i["id"] for i in row["per_recording"]}) != 178
            or any(not i["passes"] for i in row["per_recording"])
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError("Failed or incomplete V26 validation preservation gates")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved["width"] != winner["width"]:
        raise ValueError("Changed recurrence checkpoint contract")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    return plan, winner, network, (saved["mean"].numpy(), saved["scale"].numpy())


def consumed_inputs():
    # This function requires the unchanged e56554c replay on PYTHONPATH; its
    # old baseline guards must continue failing in the current website checkout.
    from release_attack_release_v25 import sealed
    from verify_attack_release_v25_runtime import all_items

    run = Path("/training/attack-release-v25-v1")
    fresh = run.parent / "attack-release-v25-fresh-v1"
    plan, winner, _, _ = sealed(run, fresh)
    rows = [i for i in all_items(run, fresh, plan, winner) if i["group"] == "test"]
    if len(rows) != 392 or len({i["id"] for i in rows}) != 392:
        raise ValueError("Incomplete V25 consumed regression coverage")
    return rows


def evaluate_run(run):
    from bass_joint_timing_v18 import margin

    plan, winner, network, normalizer = frozen(run)
    target = run / "consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed V26 regression; no retuning")
    parent = consumed_inputs()
    verifier = BassAttackDeclutter()
    items = []
    folder = run / "consumed-periodicity"
    folder.mkdir(exist_ok=False)
    observation = {
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "baseline_hashes": hashes(),
        "sources": [
            {k: i[k] for k in ("id", "audio", "audio_sha256", "cache_sha256")} for i in parent
        ],
        "no_fitting_selection_user_audio_or_inference": True,
        "feature_source_sha256": digest(Path(__file__).with_name("periodicity_features.py")),
    }
    preserve(folder / "plan.json", observation)
    records = []
    for item in parent:
        p = verifier.model.probability(
            item["frames"], item["attack_frames"], item["release_frames"], item["x"]
        )
        g = verifier.guardian.probability(item["base_x"])
        keep = verifier.model.keep(item["events"], p, g, item["duration"])
        row = {**item, "events": item["events"][keep], "base_x": item["base_x"][keep]}
        row.update(
            x=features(row["events"], row["base_x"]),
            eligible=eligible(row["events"], row["duration"]),
        )
        wave, rate = sf.read(row["audio"], dtype="float32")
        if (
            digest(Path(row["audio"])) != row["audio_sha256"]
            or rate != 22050
            or len(wave) / rate != row["duration"]
        ):
            raise ValueError("Changed consumed waveform clock")
        row["periodicity"] = recurrence_features(observe(wave, rate), row["events"])
        items.append(annotate(row))
        path = folder / (row["id"] + ".npz")
        np.savez_compressed(
            path,
            events=row["events"],
            periodicity=row["periodicity"],
            base_x=row["base_x"],
            x=row["x"],
            eligible=row["eligible"],
            plan_sha256=digest(folder / "plan.json"),
        )
        records.append(
            {
                "id": row["id"],
                "cache_sha256": digest(path),
                "parent_cache_sha256": item["cache_sha256"],
            }
        )
        if len(items) % 20 == 0:
            print(json.dumps({"observed_consumed": len(items), "total": 392}), flush=True)
    preserve(
        folder / "manifest.json",
        {
            "plan_sha256": digest(folder / "plan.json"),
            "items": records,
            "no_fitting_selection_user_audio_or_inference": True,
        },
    )
    ps = [probabilities(network, inputs(i), normalizer) for i in items]
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
        "observation_manifest_sha256": digest(folder / "manifest.json"),
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
        raise ValueError("V26 regression failed; do not generate fresh audio or export weights")
    for name in ("raw", "margin"):
        if len(report[name]["per_recording"]) != 392 or any(
            not i["passes"] for i in report[name]["per_recording"]
        ):
            raise ValueError("Incomplete V26 per-recording regression gates")
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evaluate_run(args.run)
