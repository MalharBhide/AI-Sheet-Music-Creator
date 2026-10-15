"""Reconstruct removed historical source files from hash-matched Git evidence.

Only archived code and current repository assets are copied. No dataset, upload,
saved score or production file is modified. Original reader guards stay intact.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_read(arguments):
    try:
        return subprocess.run(arguments, capture_output=True, check=False)
    except FileNotFoundError:
        # Minimal inference containers need not install Git. Missing local
        # witnesses must still fail closed, with no plausible partial reader.
        return subprocess.CompletedProcess(arguments, 127, stdout=b"", stderr=b"Git unavailable")


DEFAULT_PLAN = "training/results/release-only-v38-v1/plan.json"
EVIDENCE_NAMES = {"plan.json", "release-checks.json", "deployment.json", "checkpoint-inventory.json",
                  "batch-selection.json", "initialization.json", "preflight.json",
                  "runtime-installation-source-proof.json"}


def restore(production, output, plan_relative=DEFAULT_PLAN, full_reader=False):
    production, output = Path(production), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Preserve existing historical source tree")
    relative_plan = Path(plan_relative)
    if (relative_plan.is_absolute() or ".." in relative_plan.parts
            or relative_plan.parts[:2] != ("training", "results") or type(full_reader) is not bool):
        raise ValueError("Invalid historical baseline plan")
    plan_path = production / relative_plan
    baseline = json.loads(plan_path.read_text())["baseline_hashes"]
    expected = baseline["sources"]
    revision = baseline.get("preceding_source_commit")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        raise ValueError("Invalid historical Git revision")
    evidence = production / "training/results"
    selected, payloads = {}, {}
    for name, sha in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("backend", "training"):
            raise ValueError("Invalid historical source path")
        candidates = [production / name]
        candidates += sorted(evidence.glob("*/source-witnesses/" + name))
        candidates += sorted(evidence.glob("*/source-witnesses/" + relative.name))
        candidates += sorted(evidence.glob("*/sources/" + relative.name + ".txt"))
        match = next((p for p in candidates if p.is_file() and not p.is_symlink() and digest(p) == sha), None)
        if match is not None:
            selected[name] = str(match.relative_to(production))
            payloads[name] = match.read_bytes()
        else:
            revisions = [revision] if revision else []
            found = False
            for attempt in range(2):
                for ref in revisions:
                    if not re.fullmatch(r"[0-9a-f]{7,40}", ref):
                        raise ValueError("Invalid historical Git revision")
                    result = git_read(["git", "-C", str(production), "show", ref + ":" + name])
                    if not result.returncode and hashlib.sha256(result.stdout).hexdigest() == sha:
                        selected[name] = "git:" + ref + ":" + name
                        payloads[name] = result.stdout
                        found = True
                        break
                if found or attempt == 1:
                    break
                history = git_read(["git", "-C", str(production), "log", "--max-count=5001",
                                    "--format=%H", "--", name])
                revisions = history.stdout.decode().splitlines() if not history.returncode else []
                if len(revisions) > 5000:
                    raise ValueError("Historical source search exceeds audited revision bound")
            if not found:
                raise ValueError("No hash-matched historical source: " + name)
    # Resolve every source before creating the destination; missing historical
    # evidence must not leave a plausible but incomplete replay behind.
    if any(hashlib.sha256(data).hexdigest() != expected[name] for name, data in payloads.items()):
        raise ValueError("Historical evidence changed during restoration")
    if full_reader:
        for name, sha in baseline.get("arrays", {}).items():
            if Path(name).name != name or digest(production / "backend/app/assets" / name) != sha:
                raise ValueError("Changed inherited model array")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    if full_reader:
        ignore = shutil.ignore_patterns("results", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv")
        for name in ("backend", "training"):
            shutil.copytree(production / name, output / name, ignore=ignore)
        # Only small release/plan witnesses are needed, not large prediction
        # reports, dataset files, duplicated caches, or new symlinks.
        for path in sorted(evidence.glob("*/*.json")):
            if path.name not in EVIDENCE_NAMES or path.is_symlink():
                continue
            target = output / path.relative_to(production)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            metadata[str(path.relative_to(production))] = digest(target)
    for name, data in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = {"baseline_plan": str(relative_plan), "baseline_plan_sha256": digest(plan_path),
               "historical_source_sha256": expected,
               "source_witnesses": selected,
               "sources_restored": len(expected), "production_modified": False,
               "guards_changed": False, "no_user_audio_or_scores": True,
               "full_reader": full_reader, "copied_release_metadata_sha256": metadata}
    if str(relative_plan) == DEFAULT_PLAN:
        receipt["v38_plan_sha256"] = digest(plan_path)
    (output / "restoration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--full-reader", action="store_true")
    args = parser.parse_args()
    print(json.dumps(restore(args.production, args.output, args.plan, args.full_reader), indent=2))
