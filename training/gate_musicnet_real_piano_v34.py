"""Require a separately positive result on real piano, not only authored fixtures."""

import argparse
import json
from pathlib import Path

from evaluate_musicnet_full_v34 import frozen
from fresh_musicnet_full_v34 import regression_gate
from prepare_robust_training_stems import digest, preserve

POLICY_SHA = "912c1ebcb2753acab5b549829fe3376cfacda01c6d034c2d021fd13c3ad55c2b"
POLICY = (
    Path(__file__).resolve().parent
    / "results/musicnet-original-piano-source-v1/additional-release-policy.json"
)


def real_benefit(report, declaration):
    ids = [row["id"] for row in declaration["jobs"]]
    if len(ids) != 18 or len(set(ids)) != 18:
        raise ValueError("Incomplete predeclared real-piano first pass")
    outcomes = {}
    for name in ("raw", "margin"):
        rows = [r for r in report[name]["per_recording"] if r["id"] in ids]
        if [r["id"] for r in rows] != ids:
            raise ValueError("Changed or missing reserved real-piano identities")
        gain = 0
        for row in rows:
            before, after = row["baseline"], row["candidate"]
            if (
                row["corpus"] != "reserved-musicnet-real-piano"
                or not row["passes"]
                or after["true_positives"] < before["true_positives"]
                or after["false_positives"] > before["false_positives"]
                or not row["coverage"]["passes"]
                or not row["timing"]["passes"]
                or row["retimed_notes"]
                or any(row["matches"][kind]["lost"] for kind in ("attack", "hold"))
            ):
                raise ValueError("Failed reserved real-piano preservation")
            gain += before["false_positives"] - after["false_positives"]
        if gain <= 0:
            raise ValueError("No independently positive reserved real-piano benefit")
        outcomes[name] = {"false_notes_removed": gain, "recordings": len(rows)}
    return outcomes


def require_real_piano(run):
    if digest(POLICY) != POLICY_SHA:
        raise ValueError("Changed predeclared real-piano release policy")
    policy = json.loads(POLICY.read_text())
    plan, winner, _, _ = frozen(run)
    regression_gate(run, winner)
    declared = plan["first_pass_declaration"]
    report = json.loads((run / "original-first-pass.json").read_text())
    if (
        declared["source_manifest_sha256"] != policy["source_manifest_sha256"]
        or plan["real_piano_release_policy_sha256"] != POLICY_SHA
        or not plan["real_piano_release_requires_separate_positive_gain"]
        or not policy["declared_before_fitting"]
        or not report["passes"]
        or not report["first_pass_complete"]
        or not report["no_retuning"]
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or report["batch_selection_sha256"] != digest(run / "batch-selection.json")
    ):
        raise ValueError("Failed frozen real-piano first-pass contract")
    outcomes = real_benefit(report, declared)
    result = {
        "passes": True,
        "separate_real_piano_outcomes": outcomes,
        "policy_sha256": POLICY_SHA,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "first_pass_sha256": digest(run / "original-first-pass.json"),
        "gate_sha256": digest(Path(__file__)),
        "no_confidence_retuning_or_runner_up_selection": True,
        "new_weights_deployed": False,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    result = require_real_piano(run)
    preserve(run / "real-piano-release-gate.json", result)
    print(json.dumps(result, indent=2))
