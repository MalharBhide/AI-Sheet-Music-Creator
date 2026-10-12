"""Add measured recurrence to the existing verified public-domain fitting clips."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from current_bass_v32_baseline import hashes
from periodicity_bass_data import validate
from periodicity_features import features, observe
from prepare_mutopia_boundaries import load as source_items
from prepare_robust_training_stems import digest, preserve

VERSION = "mutopia-measured-recurrence-v42-v1"


def contracts():
    return {n: digest(Path(__file__).with_name(n)) for n in ("prepare_mutopia_recurrence_v42.py",
        "prepare_mutopia_boundaries.py", "acquire_mutopia_boundaries.py", "periodicity_features.py")}


def prepare(source, output):
    items = source_items(source)
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "source_root": str(source), "source_manifest_sha256": digest(source/"manifest.json"),
        "source_plan_sha256": digest(source/"plan.json"), "source_sha256": contracts(), "historical_v32": hashes(),
        "counts": {"train": 16, "validation": 4}, "no_decoder_inference": True, "no_user_audio_or_scores": True,
        "unchanged_events_and_labels": True, "reserved_work_untouched": True}
    preserve(output/"plan.json", plan)
    records = []
    for item in items:
        wave, rate = sf.read(item["audio"], dtype="float32")
        if rate != 22050 or wave.shape != (round(item["duration"]*rate),):
            raise ValueError("Changed original Mutopia physical clock")
        target = output/(item["id"]+".npz")
        np.savez_compressed(target, events=item["events"], periodicity=features(observe(wave, rate), item["events"]),
            parent_cache_sha256=item["cache_sha256"], plan_sha256=digest(output/"plan.json"))
        records.append({"id": item["id"], "group": item["group"], "source_group": item["source_group"],
            "parent_cache_sha256": item["cache_sha256"], "audio_sha256": item["audio_sha256"], "cache_sha256": digest(target)})
    if contracts() != plan["source_sha256"] or hashes() != plan["historical_v32"]:
        raise ValueError("Changed recurrence preparation")
    preserve(output/"manifest.json", {"version": VERSION, "items": records, "plan_sha256": digest(output/"plan.json"),
        "no_user_audio_or_scores": True, "reserved_work_untouched": True})
    print(json.dumps({"prepared": len(records), "no_decoder_inference": True}), flush=True)


def load(folder):
    plan = json.loads((folder/"plan.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    source = Path(plan["source_root"])
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["source_sha256"] != contracts() or plan["historical_v32"] != hashes()
            or plan["source_manifest_sha256"] != digest(source/"manifest.json")
            or plan["source_plan_sha256"] != digest(source/"plan.json")
            or manifest["plan_sha256"] != digest(folder/"plan.json")
            or not manifest["reserved_work_untouched"] or not manifest["no_user_audio_or_scores"]
            or len(manifest["items"]) != 20 or len({r["id"] for r in manifest["items"]}) != 20):
        raise ValueError("Changed recurrence fitting provenance")
    records = {r["id"]: r for r in manifest["items"]}
    result = []
    for item in source_items(source):
        record = records.pop(item["id"])
        path = folder/(item["id"]+".npz")
        if (any(record[k] != item[k] for k in ("group", "source_group", "audio_sha256"))
                or record["parent_cache_sha256"] != item["cache_sha256"] or digest(path) != record["cache_sha256"]):
            raise ValueError("Changed recurrence fitting identity")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["events"], item["events"])
            if str(saved["plan_sha256"]) != manifest["plan_sha256"] or str(saved["parent_cache_sha256"]) != item["cache_sha256"]:
                raise ValueError("Changed recurrence physical observations")
            item["periodicity"] = saved["periodicity"]
        result.append(validate(item))
    if records:
        raise ValueError("Extra recurrence fitting identities")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    prepare(**vars(parser.parse_args()))
