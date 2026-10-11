"""Freeze one validation winner, then check every consumed regression exactly once."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import BassHarmonic
from bass_temporal_data import annotate
from bass_training_data import eligible
from bass_v32_regression_data import load as previous_regressions
from current_bass_v32_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from release_only_v38 import VERSION, model, probabilities
from score_chord_boundary_v36 import evaluate
from train_release_only_v38 import contracts


def frozen(run):
    plan = json.loads((run / "plan.json").read_text())
    selection = json.loads((run / "batch-selection.json").read_text())
    if (plan["version"] != VERSION or plan["baseline_hashes"] != hashes()
            or plan["source_sha256"] != contracts()
            or selection["plan_sha256"] != digest(run / "plan.json")
            or not selection["selected"] or not selection["no_test_used_for_selection"]
            or not selection["no_user_audio_or_scores"]):
        raise ValueError("No frozen safe release-only winner, or changed source contract")
    for root, key in (("baseline", "baseline"), ("piano", "piano"), ("new", "new")):
        if digest(Path(plan[root+"_root"]) / "manifest.json") != plan[key+"_manifest_sha256"]:
            raise ValueError("Changed frozen fitting observations")
    winner = selection["winner"]
    profile = winner["profile"]["name"]
    path = run / winner["checkpoint"]
    validation = run / profile / "validation.json"
    if (digest(path) != winner["checkpoint_sha256"]
            or winner["validation_sha256"] != digest(validation)
            or winner != json.loads((run / profile / "selection.json").read_text())
            or winner["timing_threshold"] not in plan["timing_thresholds"]
            or winner["remove_threshold"] != 1.):
        raise ValueError("Changed frozen weights or validation witness")
    report = json.loads(validation.read_text())
    for name, threshold in (("raw", winner["timing_threshold"]),
                            ("strict", min(.9999, winner["timing_threshold"]+.03))):
        row = report[name]
        if (not row["passes"] or row["release_gain_seconds"] <= 1e-6
                or abs(row["onset_gain_seconds"]) > 1e-9 or row["false_notes_removed"] != 0
                or row["timing_threshold"] != threshold or row["remove_threshold"] != 1.
                or len(row["per_recording"]) != 225
                or len({r["id"] for r in row["per_recording"]}) != 225
                or any(not r["passes"] for r in row["per_recording"])):
            raise ValueError("Failed or incomplete release-only validation")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["version"] != VERSION or saved["width"] != winner["profile"]["width"]
            or saved["epoch"] != winner["epoch"]):
        raise ValueError("Changed checkpoint metadata")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    return plan, winner, network, tuple(saved[k].numpy() for k in ("mean", "scale"))


def consumed(plan):
    old = previous_regressions(Path(plan["regression_root"]), plan["regression_manifest_sha256"])
    folder = Path(plan["additional_consumed_root"])
    if digest(folder / "manifest.json") != plan["additional_consumed_manifest_sha256"]:
        raise ValueError("Changed additional consumed identities")
    manifest = json.loads((folder / "manifest.json").read_text())
    producer = json.loads((folder / "plan.json").read_text())
    if (not manifest["now_consumed_regression"]
            or not manifest["no_fitting_selection_user_audio_or_scores"]
            or manifest["plan_sha256"] != digest(folder / "plan.json")
            or producer["baseline_hashes"] != hashes() or len(manifest["items"]) != 50):
        raise ValueError("Incomplete consumed V34 corpus")
    for record in manifest["items"]:
        path = folder / record["id"] / "features.npz"
        if (record["group"] != "test" or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed consumed waveform or features")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed consumed observation plan")
            item = {**record, **{k: saved[k] for k in (
                "events", "x", "base_x", "attack_frames", "release_frames", "release_neighbors", "eligible")}}
        np.testing.assert_array_equal(item["eligible"], eligible(item["events"], record["duration"]))
        item.update(reference=np.asarray(record["reference"]).reshape(-1, 3),
                    pitch_reference=np.asarray(record["pitch_reference"]).reshape(-1, 3))
        old.append(annotate(item))
    if len(old) != 519 or len({i["id"] for i in old}) != 519:
        raise ValueError("Incomplete complete regression suite")
    return old


def score(items, network, normalizer, winner):
    guardian = BassHarmonic().models[1]
    ps = [probabilities(network, i, normalizer) for i in items]
    guards = [guardian.probability(i["base_x"]) for i in items]
    memo = {}
    raw = evaluate(items, ps, guards, winner["timing_threshold"], 1., memo)
    strict = evaluate(items, ps, guards, min(.9999, winner["timing_threshold"]+.03), 1., memo)
    return {"passes": all(r["passes"] and r["release_gain_seconds"] >= -1e-9
                          and abs(r["onset_gain_seconds"]) < 1e-9 and r["false_notes_removed"] == 0
                          for r in (raw, strict)), "raw": raw, "strict": strict}


def evaluate_run(run):
    target = run / "consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed regression; no runner-up or retuning")
    plan, winner, network, normalizer = frozen(run)
    torch.set_num_threads(2)
    items = consumed(plan)
    result = {**score(items, network, normalizer, winner), "recordings": len(items),
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "evaluator_sha256": digest(Path(__file__)),
        "no_test_used_for_selection": True, "no_user_audio_or_scores": True,
        "new_weights_deployed": False}
    frozen(run)
    preserve(target, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("raw", "strict")}), flush=True)


def require_regression(run):
    _, winner, _, _ = frozen(run)
    result = json.loads((run / "consumed-regression.json").read_text())
    if (not result["passes"] or result["recordings"] != 519
            or result["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or result["evaluator_sha256"] != digest(Path(__file__))
            or result["batch_selection_sha256"] != digest(run / "batch-selection.json")):
        raise ValueError("Frozen winner failed consumed regression; no deployment")
    for name in ("raw", "strict"):
        if (not result[name]["passes"] or len(result[name]["per_recording"]) != 519
                or any(not r["passes"] for r in result[name]["per_recording"])):
            raise ValueError("Incomplete regression preservation proof")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    evaluate_run(parser.parse_args().run)
