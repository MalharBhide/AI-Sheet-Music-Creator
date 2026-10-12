"""Install only the frozen, staged, native-tested V43 runtime and asset."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_witness(path, passed, skipped):
    text = path.read_text()
    if (not re.search(rf"\b{passed} passed\b", text) or re.search(r"\b(?:failed|ERROR)\b", text)
            or (skipped and not re.search(rf"\b{skipped} skipped\b", text))
            or (not skipped and re.search(r"\b\d+ skipped\b", text))):
        raise ValueError("Required complete functional test gate did not pass")
    return digest(path)


def install(directory, production):
    run, stage = directory/"left-precision-v43-v1", directory/"left-presence-v43-stage"
    replay = directory/"left-precision-v43-replay"
    plan = json.loads((run/"plan.json").read_text())
    selection = json.loads((run/"batch-selection.json").read_text())
    winner = selection["winner"]
    parity = json.loads((run/"runtime-parity.json").read_text())
    native = json.loads((run/"native-routing.json").read_text())
    staged = json.loads((stage/"stage.json").read_text())
    image = json.loads((directory/"left-presence-v43-image-check.json").read_text())
    if (not selection["selected"] or selection["plan_sha256"] != digest(run/"plan.json")
            or digest(run/winner["checkpoint"]) != winner["checkpoint_sha256"]
            or parity["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or parity["first_pass_sha256"] != digest(run/"original-first-pass.json")
            or not parity["passes"] or parity["recordings"] != 1721 or parity["physical_recordings"] != 40
            or not parity["identical_keep_decisions_both_settings"] or not parity["retained_v38_clocks_identical"]
            or parity["maximum_feature_error"] > 1e-6 or parity["maximum_probability_error"] > 1e-6
            or not native["passes"] or not native["all_fresh_deletion_cases_tested"]
            or not native["staged_module_origins_asserted"] or native["unchanged_controls"] != 2
            or native["stage_sha256"] != digest(stage/"stage.json")
            or staged["runtime_parity_sha256"] != digest(run/"runtime-parity.json")
            or native["runtime_parity_sha256"] != digest(run/"runtime-parity.json")
            or native["verifier_sha256"] != digest(production/"training/verify_left_presence_v43_native.py")
            or staged["stager_sha256"] != digest(production/"training/stage_left_presence_v43.py")
            or not image["passes"] or not all(image["dependencies"].values())
            or image["asset_sha256"] != parity["asset_sha256"] or image["threshold"] != winner["threshold"]
            or image["veto_threshold"] != winner["veto_threshold"] or not image["image_id"].startswith("sha256:")):
        raise ValueError("Incomplete or changed V43 runtime gates")
    validation = run/winner["profile"]["name"]/"validation.json"
    if digest(validation) != winner["validation_sha256"]:
        raise ValueError("Changed frozen validation")
    for filename, count in (("consumed-regression.json", 563), ("original-first-pass.json", 40)):
        report = json.loads((run/filename).read_text())
        if (not report["passes"] or report["recordings"] != count
                or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
                or report["batch_selection_sha256"] != digest(run/"batch-selection.json")
                or not report["no_user_audio_or_scores"]
                or any(not report[k]["passes"] or len(report[k]["per_recording"]) != count
                       or len({r["id"] for r in report[k]["per_recording"]}) != count
                       or any(not r["passes"] or not r["retained_clocks_identical"] for r in report[k]["per_recording"])
                       for k in ("raw", "strict"))):
            raise ValueError("Changed corpus preservation proof")
    first = json.loads((run/"original-first-pass.json").read_text())
    if (first["consumed_regression_sha256"] != digest(run/"consumed-regression.json")
            or first["test_manifest_sha256"] != digest(directory/"left-precision-v43-first-pass/manifest.json")):
        raise ValueError("Changed independent observation provenance")
    for name, expected in plan["source_sha256"].items():
        if digest(replay/"training"/name) != expected:
            raise ValueError("Frozen fitting source changed")
    release_path = production/"training/results/release-only-v38-v1/release-checks.json"
    release = json.loads(release_path.read_text())
    if digest(release_path) != plan["production"]["release_checks_sha256"]:
        raise ValueError("Approved V38 production witness changed")
    for name, expected in release["source_sha256"].items():
        if digest(production/name) != expected:
            raise ValueError("Approved production baseline changed before installation")
    for name, expected in release["arrays"].items():
        if digest(production/"backend/app/assets"/name) != expected:
            raise ValueError("Inherited model asset changed")
    for name, expected in parity["runtime_source_sha256"].items():
        if digest(production/name) != expected:
            raise ValueError("Parity checked source changed")
    for name, expected in staged["source_sha256"].items():
        if digest(stage/"backend"/name) != expected or image["source_sha256"].get(name) != expected:
            raise ValueError("Staged and candidate image sources differ")
    logs = {"staged-tests": test_witness(directory/"left-presence-v43-staged-tests.log", 986, 6),
            "native-integration": test_witness(directory/"left-presence-v43-integration.log", 6, 0)}
    backup = directory/"left-presence-v43-original-runtime"
    if backup.exists() or (run/"installation-gates.json").exists():
        raise ValueError("Preserve the original installation and rollback tree")
    backup.mkdir()
    for name in release["source_sha256"]:
        path = backup/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(production/name, path)
    for name in staged["source_sha256"]:
        shutil.copyfile(stage/"backend"/name, production/"backend"/name)
    result = {"passes": True, "source_sha256": staged["source_sha256"], "test_logs": logs,
        "runtime_parity_sha256": digest(run/"runtime-parity.json"), "native_routing_sha256": digest(run/"native-routing.json"),
        "image_check_sha256": digest(directory/"left-presence-v43-image-check.json"),
        "installer_sha256": digest(Path(__file__)), "rollback_root": str(backup),
        "website_restart_still_required": True, "previous_assets_unchanged": True,
        "original_engraving_unchanged": True, "no_user_audio_or_scores": True}
    (run/"installation-gates.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("production", type=Path)
    install(**vars(parser.parse_args()))
