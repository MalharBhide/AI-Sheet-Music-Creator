"""Evaluate the one frozen frame-presence validation winner on all consumed clips."""

import argparse
import json
from pathlib import Path

import torch
from current_bass_v32_baseline import hashes
from evaluate_left_precision_v54 import consumed
from left_precision_release_gate_v57 import require_consumed_proof, require_validation_proof
from left_precision_v44 import baseline, released_models
from left_precision_v46 import evaluate
from left_precision_v58 import VERSION, gate, model, probabilities, require_phase_contract
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v58 import (
    FOLDERS,
    PROFILES,
    SOURCES,
    STAGES,
    THRESHOLDS,
    VETO_THRESHOLDS,
    deployed_contract,
    preparation_audit,
)


def frozen(run, production):
    plan = json.loads((run/"plan.json").read_text())
    selection = json.loads((run/"batch-selection.json").read_text())
    root = Path(__file__).resolve().parent
    if (plan["version"] != VERSION or plan["production"] != deployed_contract(production)
            or plan["historical_v32"] != hashes() or not selection["selected"]
            or plan["new_preparation_audit_sha256"] != preparation_audit()
            or plan["profiles"] != list(PROFILES) or plan["hard_keep_fraction"] != .1
            or plan["source_sha256"] != {n: digest(root/n) for n in SOURCES}
            or selection["plan_sha256"] != digest(run/"plan.json") or not selection["no_test_used_for_selection"]
            or not selection["no_user_audio_or_scores"]
            or any(digest(v/"manifest.json") != plan["fitting_manifests"][k] for k, v in FOLDERS.items())):
        raise ValueError("No unchanged frozen frame-presence validation winner")
    winner = selection["winner"]
    if (winner["profile"] not in PROFILES or winner["epoch"] not in STAGES
            or winner["checkpoint"] != f'{winner["profile"]["name"]}/epoch-{winner["epoch"]:03d}.pt'):
        raise ValueError("Changed preregistered model profile or checkpoint path")
    checkpoint, validation = run/winner["checkpoint"], run/winner["profile"]["name"]/"validation.json"
    if (winner["plan_sha256"] != selection["plan_sha256"] or digest(checkpoint) != winner["checkpoint_sha256"]
            or digest(validation) != winner["validation_sha256"] or winner["threshold"] not in THRESHOLDS or winner["veto_threshold"] not in VETO_THRESHOLDS
            or winner != json.loads((validation.parent/"selection.json").read_text())):
        raise ValueError("Changed frame-presence selection or checkpoint")
    report = json.loads(validation.read_text())
    if (plan["timed_validation_recordings"] != 381 or plan["pitch_only_validation_recordings"] != 237
            or plan["training_recordings"] != 2302 or plan["validation_recordings"] != 618):
        raise ValueError("Changed timed or weak-pitch validation count")
    if (report["veto_threshold"] != winner["veto_threshold"]
            or report["stricter_veto_threshold"] != winner["veto_threshold"]*.8):
        raise ValueError("Changed separately calibrated presence policy")
    require_validation_proof(report, winner)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION or saved["width"] != winner["profile"]["width"] or saved["epoch"] != winner["epoch"]:
        raise ValueError("Changed frame-presence model metadata")
    if (any(not torch.isfinite(value).all() for value in saved["state_dict"].values())
            or any(saved[k].shape != (163,) or not torch.isfinite(saved[k]).all() for k in ("mean", "scale"))
            or torch.any(saved["scale"] < .01)):
        raise ValueError("Invalid model normalization or nonfinite learned weights")
    network = model(saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    require_phase_contract(network)
    return plan, winner, network, tuple(saved[k].numpy() for k in ("mean", "scale"))


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
    if result["passes"]:
        require_consumed_proof(result, winner, digest(run/"batch-selection.json"), digest(Path(__file__)))
    frozen(run, production)
    preserve(target, result)
    print(json.dumps({k: ({a: b for a, b in v.items() if a != "per_recording"} if k in ("raw", "strict") else v)
                      for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("production", type=Path)
    evaluate_run(**vars(parser.parse_args()))
