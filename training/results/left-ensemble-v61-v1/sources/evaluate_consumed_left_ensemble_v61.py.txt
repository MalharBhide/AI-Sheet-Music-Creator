"""Evaluate the frozen joint winner on all 603 already-consumed regressions."""

import argparse
import json
from pathlib import Path

import torch
from ensemble_contract_v61 import frozen
from evaluate_left_precision_v51 import consumed
from left_ensemble_v61 import OUTPUT_THRESHOLD, union_permissions
from left_precision_v43 import probabilities
from left_precision_v44 import baseline, released_models
from left_precision_v46 import evaluate
from prepare_robust_training_stems import digest, preserve

SOURCES = ("evaluate_consumed_left_ensemble_v61.py", "test_ensemble_regression_v61.py",
           "evaluate_left_precision_v51.py", "evaluate_left_precision_v40.py")


def regression_qualifies(report):
    if (report["recordings"] != 603 or not report["no_test_used_for_selection"]
            or not report["no_user_audio_or_scores"] or report["new_weights_deployed"]):
        return False
    identities = []
    for name in ("raw", "strict"):
        summary = report[name]
        rows = summary["per_recording"]
        identities.append({r["id"] for r in rows})
        if (len(rows) != 603 or len(identities[-1]) != 603
                or summary["threshold"] != OUTPUT_THRESHOLD or not summary["passes"]
                or summary["false_notes_removed"] <= 0 or summary["recovered_notes"] != 0
                or summary["false_notes_removed"] != sum(r["false_notes_removed"] for r in rows)
                or any(not r["passes"] or not r["retained_clocks_identical"]
                       or not r["fixed_reference_identities_preserved"] or not r["timing"]["passes"]
                       or not r["coverage"]["passes"]
                       or r["recovered_notes"] != 0 for r in rows)):
            return False
    return identities[0] == identities[1]


def evaluate_run(run, production):
    target = run/"consumed-regression.json"
    if target.exists():
        raise ValueError("Preserve completed joint regression; no runner-up or retuning")
    _, selection, models = frozen(run, production)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    sources = {n: digest(root/n) for n in SOURCES}
    print(json.dumps({"frozen_members": [m["checkpoint_sha256"] for m in selection["members"]],
                      "loading_all_603_regressions": True}), flush=True)
    items = consumed(production)
    released = released_models(production)
    heads = [[probabilities(network, i, normalizer) for i in items] for network, normalizer in models]
    memo, reports = {}, {}
    for stricter, name in ((False, "raw"), (True, "strict")):
        ps = [union_permissions(values, selection["members"], stricter) for values in zip(*heads, strict=True)]
        reports[name] = evaluate(items, baseline(items, released, stricter), ps, OUTPUT_THRESHOLD, memo)
    result = {**reports, "recordings": len(items),
              "batch_selection_sha256": digest(run/"batch-selection.json"),
              "frozen_plan_sha256": digest(run/"plan.json"), "source_sha256": sources,
              "member_checkpoint_sha256": [m["checkpoint_sha256"] for m in selection["members"]],
              "no_test_used_for_selection": True, "no_user_audio_or_scores": True, "new_weights_deployed": False}
    result["passes"] = regression_qualifies(result)
    frozen(run, production)
    if sources != {n: digest(root/n) for n in SOURCES}:
        raise ValueError("Changed joint regression producer")
    preserve(target, result)
    print(json.dumps({"complete": True, "passes": result["passes"],
                      "summary": {n: {k: v for k, v in r.items() if k != "per_recording"}
                                  for n, r in reports.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("production", type=Path)
    evaluate_run(**vars(parser.parse_args()))
