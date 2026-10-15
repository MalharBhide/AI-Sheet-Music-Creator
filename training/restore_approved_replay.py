"""Reconstruct removed historical source files from hash-matched Git evidence.

Only archived code and current repository assets are copied. No dataset, upload,
saved score or production file is modified. Original reader guards stay intact.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(production, output):
    production, output = Path(production), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Preserve existing historical source tree")
    plan_path = production / "training/results/release-only-v38-v1/plan.json"
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
        elif revision is not None:
            result = subprocess.run(["git", "-C", str(production), "show", revision + ":" + name],
                                    capture_output=True, check=False)
            if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
                raise ValueError("No hash-matched historical source: " + name)
            selected[name] = "git:" + revision + ":" + name
            payloads[name] = result.stdout
        else:
            raise ValueError("No hash-matched historical source: " + name)
    # Resolve every source before creating the destination; missing historical
    # evidence must not leave a plausible but incomplete replay behind.
    if any(hashlib.sha256(data).hexdigest() != expected[name] for name, data in payloads.items()):
        raise ValueError("Historical evidence changed during restoration")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = {"v38_plan_sha256": digest(plan_path), "historical_source_sha256": expected,
               "source_witnesses": selected,
               "sources_restored": len(expected), "production_modified": False,
               "guards_changed": False, "no_user_audio_or_scores": True}
    (output / "restoration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.production, args.output), indent=2))
