"""Evaluate the one frozen frame-presence validation winner on all consumed clips."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bass_temporal_data import annotate
from current_bass_v32_baseline import hashes
from evaluate_left_precision_v40 import consumed as previous_consumed
from left_precision_v43 import gate, model, probabilities
from left_precision_v44 import baseline, evaluate, released_models
from left_precision_v45 import VERSION
from periodicity_bass_data import validate
from prepare_left_support_v41 import KEYS
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v45 import (
    FOLDERS,
    SOURCES,
    THRESHOLDS,
    VETO_THRESHOLDS,
    deployed_contract,
)


def frozen(run, production):
    plan = json.loads((run/"plan.json").read_text())
    selection = json.loads((run/"batch-selection.json").read_text())
    root = Path(__file__).resolve().parent
    if (plan["version"] != VERSION or plan["production"] != deployed_contract(production)
            or plan["historical_v32"] != hashes() or not selection["selected"]
            or plan["source_sha256"] != {n: digest(root/n) for n in SOURCES}
            or selection["plan_sha256"] != digest(run/"plan.json") or not selection["no_test_used_for_selection"]
            or not selection["no_user_audio_or_scores"]
            or any(digest(v/"manifest.json") != plan["fitting_manifests"][k] for k, v in FOLDERS.items())):
        raise ValueError("No unchanged frozen frame-presence validation winner")
    winner = selection["winner"]
    checkpoint, validation = run/winner["checkpoint"], run/winner["profile"]["name"]/"validation.json"
    if (winner["plan_sha256"] != selection["plan_sha256"] or digest(checkpoint) != winner["checkpoint_sha256"]
            or digest(validation) != winner["validation_sha256"] or winner["threshold"] not in THRESHOLDS or winner["veto_threshold"] not in VETO_THRESHOLDS
            or winner != json.loads((validation.parent/"selection.json").read_text())):
        raise ValueError("Changed frame-presence selection or checkpoint")
    report = json.loads(validation.read_text())
    if plan["timed_validation_recordings"] != 345 or plan["pitch_only_validation_recordings"] != 237:
        raise ValueError("Changed timed or weak-pitch validation count")
    if (report["veto_threshold"] != winner["veto_threshold"]
            or report["stricter_veto_threshold"] != winner["veto_threshold"]*.8):
        raise ValueError("Changed separately calibrated presence policy")
    for name, threshold in (("raw", winner["threshold"]), ("strict", min(.9999, winner["threshold"]+.005))):
        row = report[name]
        if (not row["passes"] or row["threshold"] != threshold or row["real_piano_gain"] <= 0
                or row["false_notes_removed"] <= 0 or len(row["per_recording"]) != 345
                or len({r["id"] for r in row["per_recording"]}) != 345
                or any(not r["passes"] or not r["retained_clocks_identical"] for r in row["per_recording"])):
            raise ValueError("Failed or incomplete frame-presence preservation proof")
        pitch = report[name+"_pitch"]
        if (not pitch["passes"] or pitch["threshold"] != threshold or pitch["wrong_pitch_gain"] <= 0
                or pitch["removed_fundamental_candidates"] != 0 or pitch["timing_supervision_available"]
                or len(pitch["per_recording"]) != 237 or len({r["id"] for r in pitch["per_recording"]}) != 237
                or any(not r["passes"] or not r["retained_clocks_identical"] or r["timing_supervision_available"]
                       for r in pitch["per_recording"])):
            raise ValueError("Failed or incomplete weak-pitch preservation proof")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved["width"] != winner["profile"]["width"] or saved["epoch"] != winner["epoch"]:
        raise ValueError("Changed frame-presence model metadata")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    return plan, winner, network, tuple(saved[k].numpy() for k in ("mean", "scale"))


def consumed(production):
    items = previous_consumed(production)
    folder = Path("/training/left-precision-v43-first-pass")
    first = json.loads((production/"training/results/left-precision-v43-v1/original-first-pass.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    plan = json.loads((folder/"plan.json").read_text())
    if (len(items) != 563 or digest(folder/"manifest.json") != first["test_manifest_sha256"]
            or manifest["plan_sha256"] != digest(folder/"plan.json")
            or not manifest["now_consumed_regression"] or not manifest["no_user_audio_or_scores"]
            or plan["seeds"] != list(range(288101, 288141)) or len(manifest["items"]) != 40):
        raise ValueError("Incomplete consumed V43 first pass")
    for record in manifest["items"]:
        path = folder/record["id"]/"features.npz"
        if (record["group"] != "test" or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed consumed V43 physical observations")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed consumed V43 cache identity")
            item = {**record, **{key: saved[key] for key in KEYS}}
        item.update(reference=np.asarray(record["reference"], float).reshape(-1, 3),
                    pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3))
        items.append(annotate(validate(item)))
    if len(items) != 603 or len({i["id"] for i in items}) != 603:
        raise ValueError("Incomplete or duplicated consumed regression identities")
    return items


def evaluate_run(run, production):
    target = run/"consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed regression; no runner-up or retuning")
    _, winner, network, normalizer = frozen(run, production)
    torch.set_num_threads(2)
    print(json.dumps({"frozen_checkpoint": winner["checkpoint_sha256"], "loading_all_603_regressions": True}), flush=True)
    items = consumed(production)
    released = released_models(production)
    deployed = baseline(items, released)
    stricter_deployed = baseline(items, released, stricter=True)
    heads = [probabilities(network, i, normalizer) for i in items]
    ps = [gate(h, winner["veto_threshold"]) for h in heads]
    stricter_ps = [gate(h, winner["veto_threshold"]*.8) for h in heads]
    memo = {}
    raw = evaluate(items, deployed, ps, winner["threshold"], memo)
    strict = evaluate(items, stricter_deployed, stricter_ps, min(.9999, winner["threshold"]+.005), memo)
    result = {"passes": raw["passes"] and strict["passes"] and raw["false_notes_removed"] > 0 and strict["false_notes_removed"] > 0, "raw": raw, "strict": strict,
        "recordings": len(items), "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run/"batch-selection.json"), "evaluator_sha256": digest(Path(__file__)),
        "no_test_used_for_selection": True, "no_user_audio_or_scores": True, "new_weights_deployed": False}
    frozen(run, production)
    preserve(target, result)
    print(json.dumps({k: ({a: b for a, b in v.items() if a != "per_recording"} if k in ("raw", "strict") else v)
                      for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("production", type=Path)
    evaluate_run(**vars(parser.parse_args()))
