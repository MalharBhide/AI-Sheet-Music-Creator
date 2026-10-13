"""Evaluate the one preregistered trained pair on complete validation only."""

import argparse
import json
from pathlib import Path

import torch
from current_bass_v32_baseline import hashes
from ensemble_contract_v56 import (
    SOURCES,
    choose_member,
    load_member,
    protocol,
    validation_qualifies,
)
from left_ensemble_v56 import OUTPUT_THRESHOLD, VERSION, union_permissions
from left_precision_v43 import probabilities
from left_precision_v44 import baseline, deployed_contract, released_models
from left_precision_v46 import evaluate, pitch_evaluate
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v54 import fitting_items


def evaluate_run(production, output):
    if output.exists():
        raise ValueError("Preserve the fixed-pair first evaluation; no runner-up or retuning")
    torch.set_num_threads(2)
    registered, receipt = protocol(production)
    members = [choose_member(production, registered["source_studies"][study], study, role)
               for study, role in (("v53", "A"), ("v54", "B"))]
    models = [load_member(m) for m in members]
    root = Path(__file__).resolve().parent
    plan = {"version": VERSION, "protocol": receipt, "members": members,
            "production": deployed_contract(production), "historical_v32": hashes(),
            "source_sha256": {n: digest(root/n) for n in SOURCES},
            "validation_recordings": 618, "timed_validation_recordings": 381,
            "pitch_only_validation_recordings": 237, "output_threshold": OUTPUT_THRESHOLD,
            "no_new_fitting_targets": True, "no_consumed_or_fresh_tests_read": True,
            "no_user_audio_or_scores": True, "new_weights_deployed": False}
    output.mkdir()
    preserve(output/"plan.json", plan)
    print(json.dumps({"fixed_members": [{k: m[k] for k in ("study", "profile", "epoch", "threshold", "veto_threshold")}
                                         for m in members], "loading_validation_only_for_joint_scoring": True}), flush=True)
    # Source loaders validate all existing fitting manifests. Only validation
    # observations are retained; no new inference or optimization is performed.
    validation = [i for i in fitting_items() if i["group"] == "validation"]
    timed = [i for i in validation if i["corpus"] != "tinysol-isolated-pitch"]
    isolated = [i for i in validation if i["corpus"] == "tinysol-isolated-pitch"]
    if len(timed) != 381 or len(isolated) != 237 or len({i["id"] for i in validation}) != 618:
        raise ValueError("Incomplete joint validation population")
    released, reports, memo = released_models(production), {}, {}
    heads = {family: [[probabilities(network, i, normalizer) for i in group]
                      for network, normalizer in models]
             for family, group in (("timed", timed), ("pitch", isolated))}
    for stricter, name in ((False, "raw"), (True, "strict")):
        ps = {family: [union_permissions(values, members, stricter)
                       for values in zip(*values_by_model, strict=True)]
              for family, values_by_model in heads.items()}
        reports[name] = evaluate(timed, baseline(timed, released, stricter), ps["timed"], OUTPUT_THRESHOLD, memo)
        reports[name]["nrgcp_gain"] = sum(r["false_notes_removed"] for r in reports[name]["per_recording"]
                                        if r["corpus"] == "nrgcp-authored-sampled-piano")
        reports[name+"_pitch"] = pitch_evaluate(isolated, baseline(isolated, released, stricter), ps["pitch"], OUTPUT_THRESHOLD)
    if (plan["production"] != deployed_contract(production) or plan["historical_v32"] != hashes()
            or plan["source_sha256"] != {n: digest(root/n) for n in SOURCES}
            or members != [choose_member(production, registered["source_studies"][study], study, role)
                           for study, role in (("v53", "A"), ("v54", "B"))]):
        raise ValueError("Fixed ensemble sources changed during evaluation")
    preserve(output/"validation.json", reports)
    passed = validation_qualifies(reports)
    selection = {"selected": passed, "members": members, "plan_sha256": digest(output/"plan.json"),
                 "validation_sha256": digest(output/"validation.json"), "no_test_used_for_selection": True,
                 "no_user_audio_or_scores": True, "new_weights_deployed": False}
    preserve(output/"batch-selection.json", selection)
    print(json.dumps({"complete": True, "selected": passed,
                      "summary": {n: {k: v for k, v in r.items() if k != "per_recording"}
                                  for n, r in reports.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    evaluate_run(**vars(parser.parse_args()))
