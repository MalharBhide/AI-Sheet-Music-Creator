"""Freeze a validation winner, then evaluate all 563 consumed recordings once."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.services.chord_release import ReleaseModel
from app.services.waveform_recurrence import features as recurrence
from bass_temporal_data import annotate
from current_bass_v32_baseline import hashes
from evaluate_release_only_v38 import consumed as old_consumed
from left_precision_v40 import VERSION, model, probabilities
from prepare_robust_training_stems import digest, preserve
from score_left_precision_v39 import baseline, evaluate
from train_left_precision_v40 import SOURCES, deployed_contract


def frozen(run, production):
    plan = json.loads((run / "plan.json").read_text())
    selection = json.loads((run / "batch-selection.json").read_text())
    root = Path(__file__).resolve().parent
    if (plan["version"] != VERSION or plan["production"] != deployed_contract(production)
            or plan["historical_v32"] != hashes()
            or plan["source_sha256"] != {name: digest(root / name) for name in SOURCES}
            or selection["plan_sha256"] != digest(run / "plan.json")
            or not selection["selected"] or not selection["no_test_used_for_selection"]
            or not selection["no_user_audio_or_scores"]):
        raise ValueError("No unchanged frozen validation winner")
    folders = {"baseline": "/training/bass-v32-baseline-v1", "piano": "/training/musicnet-v32-observations-v1"}
    if any(digest(Path(folder) / "manifest.json") != plan["fitting_manifests"][key]
           for key, folder in folders.items()):
        raise ValueError("Changed fitting observations")
    winner = selection["winner"]
    checkpoint = run / winner["checkpoint"]
    validation = run / winner["profile"]["name"] / "validation.json"
    if (winner["plan_sha256"] != selection["plan_sha256"]
            or digest(checkpoint) != winner["checkpoint_sha256"]
            or digest(validation) != winner["validation_sha256"]
            or winner["threshold"] not in plan["thresholds"]
            or winner != json.loads((validation.parent / "selection.json").read_text())):
        raise ValueError("Changed selection or checkpoint")
    report = json.loads(validation.read_text())
    for name, threshold in (("raw", winner["threshold"]), ("strict", min(.9999, winner["threshold"]+.005))):
        row = report[name]
        if (not row["passes"] or row["threshold"] != threshold or row["real_piano_gain"] <= 0
                or row["false_notes_removed"] <= 0 or len(row["per_recording"]) != 221
                or len({r["id"] for r in row["per_recording"]}) != 221
                or any(not r["passes"] or not r["retained_clocks_identical"] for r in row["per_recording"])):
            raise ValueError("Failed or incomplete validation preservation proof")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["version"] != VERSION or saved["width"] != winner["profile"]["width"]
            or saved["epoch"] != winner["epoch"]):
        raise ValueError("Changed model metadata")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    return plan, winner, network, tuple(saved[k].numpy() for k in ("mean", "scale"))


def consumed(production):
    old_plan = json.loads((production / "training/results/release-only-v38-v1/plan.json").read_text())
    items = old_consumed(old_plan)
    # These arrays already exist in the verified V34 caches; no waveform inference.
    folder = Path(old_plan["additional_consumed_root"])
    records = {r["id"]: r for r in json.loads((folder / "manifest.json").read_text())["items"]}
    for item in items:
        if item["id"] in records:
            path = folder / item["id"] / "features.npz"
            if digest(path) != item["cache_sha256"]:
                raise ValueError("Changed consumed recurrence windows")
            with np.load(path, allow_pickle=False) as saved:
                item.update({k: saved[k] for k in ("frames", "periodicity")})
    folder = Path("/training/release-only-v38-first-pass")
    first = json.loads((production / "training/results/release-only-v38-v1/original-first-pass.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (digest(folder / "manifest.json") != first["test_manifest_sha256"]
            or manifest["plan_sha256"] != digest(folder / "plan.json")
            or not manifest["now_consumed_regression"] or not manifest["no_user_audio_or_scores"]
            or len(manifest["items"]) != 44 or len({r["id"] for r in manifest["items"]}) != 44):
        raise ValueError("Incomplete consumed V38 first pass")
    for record in manifest["items"]:
        path = folder / record["id"] / "features.npz"
        if (record["group"] != "test" or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed consumed V38 waveform or observation")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed consumed V38 feature identity")
            item = {**record, **{k: saved[k] for k in ("events", "base_x", "x", "eligible", "frames",
                "attack_frames", "release_frames", "release_neighbors")}}
        wave, rate = sf.read(record["audio"], dtype="float32")
        item["periodicity"] = recurrence(wave, rate, item["events"])
        item.update(reference=np.asarray(record["reference"]).reshape(-1, 3),
                    pitch_reference=np.asarray(record["pitch_reference"]).reshape(-1, 3))
        items.append(annotate(item))
    if len(items) != 563 or len({i["id"] for i in items}) != 563:
        raise ValueError("Incomplete consumed identities")
    return items


def score(items, production, network, normalizer, winner):
    with np.load(production / "backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        deployed = baseline(items, ReleaseModel(saved))
    ps = [probabilities(network, item, normalizer) for item in items]
    memo = {}
    raw = evaluate(items, deployed, ps, winner["threshold"], memo)
    strict = evaluate(items, deployed, ps, min(.9999, winner["threshold"]+.005), memo)
    return {"passes": raw["passes"] and strict["passes"], "raw": raw, "strict": strict}


def evaluate_run(run, production):
    target = run / "consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed regression; no runner-up or retuning")
    _, winner, network, normalizer = frozen(run, production)
    torch.set_num_threads(2)
    print(json.dumps({"frozen_checkpoint": winner["checkpoint_sha256"], "loading_all_563_regressions": True}), flush=True)
    items = consumed(production)
    result = {**score(items, production, network, normalizer, winner), "recordings": len(items),
        "checkpoint_sha256": winner["checkpoint_sha256"], "batch_selection_sha256": digest(run / "batch-selection.json"),
        "evaluator_sha256": digest(Path(__file__)), "no_test_used_for_selection": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    frozen(run, production)
    preserve(target, result)
    print(json.dumps({"passes": result["passes"], "recordings": len(items),
        "raw": {k: v for k, v in result["raw"].items() if k != "per_recording"},
        "strict": {k: v for k, v in result["strict"].items() if k != "per_recording"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("production", type=Path)
    evaluate_run(**vars(parser.parse_args()))
