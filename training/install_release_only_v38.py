"""Install only the empirically checked, staged and native-tested runtime files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(directory):
    root = Path(__file__).resolve().parents[1]
    run = directory / "release-only-v38-v1"
    stage = directory / "release-only-v38-stage"
    plan = json.loads((run / "plan.json").read_text())
    selection = json.loads((run / "batch-selection.json").read_text())
    winner = selection["winner"]
    parity = json.loads((run / "runtime-parity.json").read_text())
    native = json.loads((run / "native-routing.json").read_text())
    staged = json.loads((stage / "stage.json").read_text())
    image = json.loads((directory / "release-only-v38-image-check.json").read_text())
    if (not selection["selected"] or selection["plan_sha256"] != digest(run / "plan.json")
            or winner["checkpoint_sha256"] != digest(run / winner["checkpoint"])
            or parity["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or not parity["passes"] or parity["recordings"] != 1569 or parity["physical_recordings"] != 44
            or not native["passes"] or not native["staged_module_origins_asserted"]
            or not native["all_fresh_corrected_cases_tested"]
            or native["stage_sha256"] != digest(stage / "stage.json")
            or staged["runtime_parity_sha256"] != digest(run / "runtime-parity.json")
            or native["runtime_parity_sha256"] != digest(run / "runtime-parity.json")
            or native["verifier_sha256"] != digest(root / "training/verify_release_only_v38_native.py")
            or not image["passes"] or image["asset_sha256"] != parity["asset_sha256"]):
        raise ValueError("Incomplete or changed release gates")
    witnesses = {}
    for filename, count in (("release-only-v38-staged-tests-repaired.log", 947),
                            ("release-only-v38-native-integration.log", 6)):
        path = directory / filename
        text = path.read_text()
        if f"{count} passed" not in text or " failed" in text or "ERROR" in text:
            raise ValueError("Required functional test suite did not pass")
        witnesses[filename] = digest(path)
    for name, expected in plan["baseline_hashes"]["sources"].items():
        if digest(root / name) != expected:
            raise ValueError("Approved baseline changed before installation")
    for name, expected in plan["baseline_hashes"]["arrays"].items():
        if digest(root / "backend/app/assets" / name) != expected:
            raise ValueError("Inherited model changed")
    for name, expected in parity["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Parity checked implementation changed")
    for name, expected in staged["source_sha256"].items():
        if digest(stage / "backend" / name) != expected or image["source_sha256"][name] != expected:
            raise ValueError("Staged/runtime image sources differ")
    for filename, count in (("consumed-regression.json", 519), ("original-first-pass.json", 44)):
        result = json.loads((run / filename).read_text())
        if (not result["passes"] or result["recordings"] != count
                or result["checkpoint_sha256"] != winner["checkpoint_sha256"]
                or result["batch_selection_sha256"] != digest(run / "batch-selection.json")
                or any(not result[k]["passes"] or len(result[k]["per_recording"]) != count
                       or any(not r["passes"] for r in result[k]["per_recording"])
                       for k in ("raw", "strict"))):
            raise ValueError("Failed or incomplete corpus preservation")
    backup = directory / "release-only-v38-original-runtime"
    backup.mkdir(exist_ok=False)
    # Preserve original strict baseline sources for auditable historical replay.
    for name in plan["baseline_hashes"]["sources"]:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    for name in staged["source_sha256"]:
        shutil.copyfile(stage / "backend" / name, root / "backend" / name)
    (run / "installation-gates.json").write_text(json.dumps({
        "passes": True, "source_sha256": staged["source_sha256"], "test_logs": witnesses,
        "runtime_parity_sha256": digest(run / "runtime-parity.json"),
        "native_routing_sha256": digest(run / "native-routing.json"),
        "image_check_sha256": digest(directory / "release-only-v38-image-check.json"),
        "installer_sha256": digest(Path(__file__)), "website_restart_still_required": True,
        "no_user_audio_or_scores": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    install(parser.parse_args().directory)
