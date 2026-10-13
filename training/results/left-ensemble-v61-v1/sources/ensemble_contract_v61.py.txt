"""Verify fixed ensemble members and complete joint validation before test access."""

import json
from pathlib import Path

import numpy as np
import torch
from current_bass_v32_baseline import hashes
from left_ensemble_v61 import OUTPUT_THRESHOLD, VERSION, select_member, selection_key
from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v44 import deployed_contract
from left_precision_v58 import VERSION as V58
from left_precision_v58 import model as model58
from left_precision_v58 import require_phase_contract
from left_precision_v59 import VERSION as V59
from left_precision_v59 import model as model59
from prepare_robust_training_stems import digest
from train_left_precision_v58 import PROFILES as PROFILES58
from train_left_precision_v59 import PROFILES as PROFILES59

SOURCES = ("left_ensemble_v61.py", "ensemble_contract_v61.py", "evaluate_left_ensemble_v61.py",
           "test_left_ensemble_v61.py", "test_ensemble_contract_v61.py",
           "evaluate_consumed_left_ensemble_v61.py", "test_ensemble_regression_v61.py",
           "fresh_left_ensemble_v61.py", "left_ensemble_v56.py")
STUDIES = {"v58": (V58, PROFILES58, model58), "v59": (V59, PROFILES59, model59)}


def protocol(production):
    folder = production/"training/results/left-ensemble-v61-v1"
    path = folder/"preregistration.json"
    plan = json.loads(path.read_text())
    root = Path(__file__).resolve().parent
    if (plan["version"] != VERSION or plan["combined_output_threshold"] != OUTPUT_THRESHOLD
            or not plan["no_joint_predictions_before_registration"]
            or not plan["no_consumed_or_fresh_tests_read"]
            or plan["source_sha256"] != {n: digest(root/n) for n in SOURCES}
            or plan["approved_asset_sha256"] != deployed_contract(production)["presence_asset_sha256"]
            or set(plan["source_studies"]) != set(STUDIES)):
        raise ValueError("Changed preregistered joint-study protocol")
    return plan, {"preregistration_sha256": digest(path)}


def choose_member(production, declared, study, role):
    version, profiles, _ = STUDIES[study]
    run = Path(declared["run"])
    if str(run) != f"/training/left-precision-{study}-v1":
        raise ValueError("Unregistered member source")
    plan = json.loads((run/"plan.json").read_text())
    batch = json.loads((run/"batch-selection.json").read_text())
    archive = production/f"training/results/left-precision-{study}-v1"
    result = json.loads((archive/"result.json").read_text())
    if (digest(run/"plan.json") != declared["plan_sha256"] or plan["version"] != version
            or plan["profiles"] != list(profiles) or plan["epochs"] != 100
            or plan["stages"] != [25, 50, 100]
            or plan["production"] != deployed_contract(production) or plan["historical_v32"] != hashes()
            or batch["selected"] is not False or batch["winner"] is not None
            or batch["plan_sha256"] != declared["plan_sha256"]
            or result["status"] != "withheld_no_validation_winner"
            or result["batch_selection_sha256"] != digest(run/"batch-selection.json")
            or not result["no_consumed_or_fresh_tests_read"]
            or (run/"consumed-regression.json").exists() or (run/"original-first-pass.json").exists()
            or any(digest(production/"training"/n) != sha for n, sha in plan["source_sha256"].items())
            or any(digest(Path(p)/"manifest.json") != plan["fitting_manifests"][k]
                   for k, p in _folders(study).items())
            or (plan["training_recordings"], plan["timed_validation_recordings"],
                plan["pitch_only_validation_recordings"]) != (2302, 381, 237)):
        raise ValueError("Changed or incomplete source-study witness")
    candidates = []
    search_hashes = {}
    for index, profile in enumerate(profiles):
        directory = run/profile["name"]
        curves = json.loads((directory/"learning-curves.json").read_text())
        if [c["epoch"] for c in curves] != list(range(1, 101)):
            raise ValueError("Incomplete member fitting")
        for epoch in plan["stages"]:
            relative = f"{profile['name']}/search-{epoch:03d}.json"
            path = run/relative
            if digest(path) != digest(archive/relative):
                raise ValueError("Changed archived member validation")
            rows = json.loads(path.read_text())
            if (len(rows) != len(plan["thresholds"])*len(plan["veto_thresholds"])
                    or {(r["raw"]["threshold"], r["veto_threshold"]) for r in rows}
                    != {(t, v) for t in plan["thresholds"] for v in plan["veto_thresholds"]}):
                raise ValueError("Incomplete member calibration")
            search_hashes[relative] = digest(path)
            candidates.extend({"row": r, "epoch": epoch, "profile_index": index,
                               "profile": dict(profile)} for r in rows)
    chosen = select_member(candidates, role)
    row = chosen["row"]
    checkpoint = f"{chosen['profile']['name']}/epoch-{chosen['epoch']:03d}.pt"
    if digest(run/checkpoint) != result["checkpoints_local_only"][checkpoint]:
        raise ValueError("Changed selected member checkpoint")
    return {"role": role, "study": study, "run": str(run), "version": version,
            "profile": chosen["profile"], "epoch": chosen["epoch"],
            "threshold": row["raw"]["threshold"], "veto_threshold": row["veto_threshold"],
            "checkpoint": checkpoint, "checkpoint_sha256": digest(run/checkpoint),
            "source_plan_sha256": declared["plan_sha256"], "source_result_sha256": digest(archive/"result.json"),
            "search_sha256": search_hashes, "selection_key": list(selection_key(chosen, role))}


def _folders(study):
    from train_left_precision_v58 import FOLDERS as folders58
    from train_left_precision_v59 import FOLDERS as folders59

    return folders58 if study == "v58" else folders59


def load_member(member):
    path = Path(member["run"])/member["checkpoint"]
    if digest(path) != member["checkpoint_sha256"]:
        raise ValueError("Changed frozen member checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["version"] != member["version"] or saved["epoch"] != member["epoch"]
            or saved["width"] != member["profile"]["width"]
            or any(v.dtype != torch.float32 or not torch.isfinite(v).all()
                   for v in saved["state_dict"].values())):
        raise ValueError("Changed or nonfinite member parameters")
    normalizer = tuple(saved[k].numpy() for k in ("mean", "scale"))
    if any(v.shape != (CONTEXT_WIDTH,) or v.dtype != np.float32 or not np.isfinite(v).all()
           for v in normalizer) or np.any(normalizer[1] <= 0):
        raise ValueError("Invalid member normalizer")
    network = STUDIES[member["study"]][2](saved["width"])
    network.load_state_dict(saved["state_dict"], strict=True)
    require_phase_contract(network)
    network.eval()
    return network, normalizer


def validation_qualifies(report):
    for name in ("raw", "strict"):
        timed, pitch = report[name], report[name+"_pitch"]
        rows, isolated = timed["per_recording"], pitch["per_recording"]
        if (len(rows) != 381 or len({r["id"] for r in rows}) != 381
                or len(isolated) != 237 or len({r["id"] for r in isolated}) != 237
                or timed["threshold"] != OUTPUT_THRESHOLD or pitch["threshold"] != OUTPUT_THRESHOLD
                or not timed["passes"] or not pitch["passes"]
                or timed["false_notes_removed"] <= 0 or timed["real_piano_gain"] <= 0
                or timed["nrgcp_gain"] <= 0 or pitch["wrong_pitch_gain"] <= 0
                or pitch["timing_supervision_available"] or pitch["removed_fundamental_candidates"] != 0
                or sum(r["corpus"] == "musicnet-real-piano" for r in rows) != 15
                or sum(r["corpus"] == "nrgcp-authored-sampled-piano" for r in rows) != 36
                or timed["false_notes_removed"] != sum(r["false_notes_removed"] for r in rows)
                or timed["real_piano_gain"] != sum(r["false_notes_removed"] for r in rows if r["corpus"] == "musicnet-real-piano")
                or timed["nrgcp_gain"] != sum(r["false_notes_removed"] for r in rows if r["corpus"] == "nrgcp-authored-sampled-piano")
                or pitch["wrong_pitch_gain"] != sum(r["wrong_pitch_gain"] for r in isolated)
                or any(not r["passes"] or not r["fixed_reference_identities_preserved"]
                       or not r["retained_clocks_identical"] or not r["timing"]["passes"] or not r["coverage"]["passes"]
                       or r["recovered_notes"] != 0 for r in rows)
                or any(not r["passes"] or not r["retained_clocks_identical"] or r["recovered_notes"] != 0
                       or r["removed_fundamental_candidates"] != 0 or r["timing_supervision_available"]
                       or r["corpus"] != "tinysol-isolated-pitch" for r in isolated)):
            return False
    return all({r["id"] for r in report["raw"+suffix]["per_recording"]}
               == {r["id"] for r in report["strict"+suffix]["per_recording"]}
               for suffix in ("", "_pitch"))


def frozen(run, production):
    registered, receipt = protocol(production)
    plan, selection = [json.loads((run/n).read_text()) for n in ("plan.json", "batch-selection.json")]
    root = Path(__file__).resolve().parent
    if (selection["selected"] is not True or selection["plan_sha256"] != digest(run/"plan.json")
            or plan["version"] != VERSION or plan["protocol"] != receipt
            or plan["production"] != deployed_contract(production) or plan["historical_v32"] != hashes()
            or plan["source_sha256"] != {n: digest(root/n) for n in SOURCES}
            or not selection["no_test_used_for_selection"] or not selection["no_user_audio_or_scores"]
            or selection["validation_sha256"] != digest(run/"validation.json")
            or not validation_qualifies(json.loads((run/"validation.json").read_text()))):
        raise ValueError("No unchanged qualifying joint validation winner")
    members = [choose_member(production, registered["source_studies"][study], study, role)
               for study, role in (("v58", "A"), ("v59", "B"))]
    if members != plan["members"] or members != selection["members"]:
        raise ValueError("Changed fixed ensemble member choice")
    return plan, selection, [load_member(m) for m in members]
