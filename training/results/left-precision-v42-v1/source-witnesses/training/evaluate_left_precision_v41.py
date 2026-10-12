"""Evaluate the one frozen dual-head validation winner on all consumed clips."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.chord_release import ReleaseModel
from current_bass_v32_baseline import hashes
from evaluate_left_precision_v40 import consumed
from left_precision_v41 import VERSION, model, probabilities
from prepare_robust_training_stems import digest, preserve
from score_left_precision_v39 import baseline, evaluate
from train_left_precision_v41 import FOLDERS, SOURCES, THRESHOLDS, deployed_contract


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
        raise ValueError("No unchanged frozen dual-head validation winner")
    winner = selection["winner"]
    checkpoint, validation = run/winner["checkpoint"], run/winner["profile"]["name"]/"validation.json"
    if (winner["plan_sha256"] != selection["plan_sha256"] or digest(checkpoint) != winner["checkpoint_sha256"]
            or digest(validation) != winner["validation_sha256"] or winner["threshold"] not in THRESHOLDS
            or winner != json.loads((validation.parent/"selection.json").read_text())):
        raise ValueError("Changed dual-head selection or checkpoint")
    report = json.loads(validation.read_text())
    for name, threshold in (("raw", winner["threshold"]), ("strict", min(.9999, winner["threshold"]+.005))):
        row = report[name]
        if (not row["passes"] or row["threshold"] != threshold or row["real_piano_gain"] <= 0
                or row["false_notes_removed"] <= 0 or len(row["per_recording"]) != 245
                or len({r["id"] for r in row["per_recording"]}) != 245
                or any(not r["passes"] or not r["retained_clocks_identical"] for r in row["per_recording"])):
            raise ValueError("Failed or incomplete dual-head preservation proof")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved["width"] != winner["profile"]["width"] or saved["epoch"] != winner["epoch"]:
        raise ValueError("Changed dual-head model metadata")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    return plan, winner, network, tuple(saved[k].numpy() for k in ("mean", "scale"))


def evaluate_run(run, production):
    target = run/"consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed regression; no runner-up or retuning")
    _, winner, network, normalizer = frozen(run, production)
    torch.set_num_threads(2)
    print(json.dumps({"frozen_checkpoint": winner["checkpoint_sha256"], "loading_all_563_regressions": True}), flush=True)
    items = consumed(production)
    with np.load(production/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        deployed = baseline(items, ReleaseModel(saved))
    ps = [probabilities(network, i, normalizer) for i in items]
    memo = {}
    raw = evaluate(items, deployed, ps, winner["threshold"], memo)
    strict = evaluate(items, deployed, ps, min(.9999, winner["threshold"]+.005), memo)
    result = {"passes": raw["passes"] and strict["passes"], "raw": raw, "strict": strict,
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
