"""Observe training/validation waveforms on the explicit V25 baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from bass_training_data import identity
from bass_v25_data import load
from current_bass_v25_baseline import hashes
from periodicity_features import NAMES, VERSION, features, observe
from prepare_robust_training_stems import digest, preserve


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/periodicity_features.py",
        "training/prepare_periodicity_bass.py",
        "training/bass_v25_data.py",
        "training/current_bass_v25_baseline.py",
    )
    return {name: digest(root / name) for name in names}


def prepare(parent, output):
    items = load(parent)
    if len(items) != 733 or any(i["group"] not in ("train", "validation") for i in items):
        raise ValueError("Periodicity fitting must exclude all regressions")
    plan = {
        "version": VERSION,
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "parent_root": str(parent),
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "parent_plan_sha256": digest(parent / "plan.json"),
        "feature_names": list(NAMES),
        "no_inference_or_user_audio": True,
        "no_regression_audio_or_metrics": True,
    }
    if output.exists():
        if (
            json.loads((output / "plan.json").read_text()) != plan
            or (output / "manifest.json").exists()
        ):
            raise ValueError("Preserve complete or changed periodicity preparation")
    else:
        output.mkdir()
        preserve(output / "plan.json", plan)
    folder = output / "features"
    folder.mkdir(exist_ok=True)
    records = []
    for item in items:
        record = {
            k: item[k] for k in ("id", "source_group", "group", "audio_sha256", "cache_sha256")
        }
        record["parent_cache_sha256"] = record.pop("cache_sha256")
        path = folder / (item["id"] + ".npz")
        completion = folder / (item["id"] + ".json")
        if completion.exists():
            complete = json.loads(completion.read_text())
            if complete["record"] != {**record, "cache_sha256": digest(path)} or complete[
                "plan_sha256"
            ] != digest(output / "plan.json"):
                raise ValueError("Changed periodicity record")
            record = complete["record"]
        else:
            waveform, rate = sf.read(item["audio"], dtype="float32")
            if (
                len(waveform) / rate != item["duration"]
                or digest(Path(item["audio"])) != item["audio_sha256"]
            ):
                raise ValueError("Changed periodicity source clock")
            values = features(observe(waveform, rate), item["events"])
            np.savez_compressed(
                path,
                events=item["events"],
                periodicity=values,
                identity=identity(record),
                plan_sha256=digest(output / "plan.json"),
            )
            record["cache_sha256"] = digest(path)
            preserve(completion, {"record": record, "plan_sha256": digest(output / "plan.json")})
        records.append(record)
        if len(records) % 20 == 0:
            print(json.dumps({"cached": len(records), "total": len(items)}), flush=True)
    if plan["baseline_hashes"] != hashes() or plan["code_sha256"] != contracts():
        raise ValueError("Periodicity implementation or baseline changed during observation")
    preserve(
        output / "manifest.json",
        {
            "version": VERSION,
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "no_regression_audio_or_metrics": True,
            "no_inference_or_user_audio": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "recordings": len(records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.output)
