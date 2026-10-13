"""Strictly audit new generated piano observations without audio inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from keynata_piano_source_v57 import WINDOWS
from left_precision_v57 import frame_supervision
from offline_piano_v47 import RATE
from prepare_keynata_piano_v57 import load, native_receipt, sources, tasks
from prepare_robust_training_stems import digest, preserve


def audit(folder, output):
    plan = json.loads((folder/"plan.json").read_text())
    source = Path(plan["source_root"])
    native = native_receipt(Path(plan["native_preflight"]), source, tasks(sources(source)))
    items = load(folder)
    rows = []
    for item in items:
        before = native[item["id"]]
        wave, rate = sf.read(folder/item["id"]/"original.wav", dtype="float32")
        if (rate != RATE or wave.shape != (24*RATE,)
                or hashlib.sha256(wave.tobytes()).hexdigest() != before["wave_samples_sha256"]
                or len(item["reference"]) != before["keys"]):
            raise ValueError("Changed native samples or authored reference keys")
        target, weight = frame_supervision(item)
        if np.any(weight[target > 0]) or np.any(weight[target == 0] != 1):
            raise ValueError("Invented generated piano acoustic-presence targets")
        selected = item["eligible"] & item["mask"]
        rows.append({"id": item["id"], "source_group": item["source_group"],
            "candidate_notes": len(item["events"]),
            "supervised_keep": int((item["y"][selected] == 1).sum()),
            "supervised_remove": int((item["y"][selected] == 0).sum()),
            "complete_reference_keys": len(item["reference"]),
            "unknown_positive_frames": int((weight == 0).sum()),
            "known_absent_frames": int(weight.sum()),
            "codec_lag_samples": item["codec_clock"]["codec_lag_samples"],
            "padding_adjustment_samples": item["codec_clock"]["silent_padding_adjustment_samples"],
            "native_sample_values_bit_identical": True})
    expected = {"keynata-generated/"+name: count*3 for name, count in WINDOWS.items()}
    if (len(items) != 72 or len({i["id"] for i in items}) != 72
            or dict(Counter(i["source_group"] for i in items)) != expected
            or any(i["group"] != "train" for i in items)
            or any(row["codec_lag_samples"] != 0 for row in rows)):
        raise ValueError("Incomplete or leaking new parent partition or MP3 clock")
    totals = {key: sum(row[key] for row in rows) for key in (
        "candidate_notes", "supervised_keep", "supervised_remove", "complete_reference_keys",
        "unknown_positive_frames", "known_absent_frames")}
    result = {"version": "keynata-piano-v57-strict-preparation-audit-v1", "recordings": 72,
        "training_parent_groups": 5, **totals, "manifest_sha256": digest(folder/"manifest.json"),
        "plan_sha256": digest(folder/"plan.json"), "native_preflight_sha256": digest(Path(plan["native_preflight"])),
        "audit_producer_sha256": digest(Path(__file__)), "no_model_inference": True,
        "no_user_audio_or_scores": True, "no_consumed_or_fresh_tests_read": True,
        "new_weights_deployed": False, "all_native_samples_bit_identical": True,
        "all_codec_lags_zero": True, "generated_positive_support_frames_masked": True,
        "per_recording": rows}
    preserve(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_recording"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    audit(**vars(parser.parse_args()))
