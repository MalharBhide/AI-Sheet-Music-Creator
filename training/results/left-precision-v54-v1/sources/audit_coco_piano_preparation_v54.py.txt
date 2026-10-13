"""Audit completed generated piano observations without repeating inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from left_precision_v54 import frame_supervision
from offline_piano_v47 import RATE
from prepare_coco_piano_v54 import contracts, load
from prepare_robust_training_stems import digest, preserve


def audit(folder, preflight, output):
    native = json.loads(preflight.read_text())
    if (native["code_sha256"] != contracts() or not native["no_model_inference"]
            or not native["no_user_audio_or_scores"]):
        raise ValueError("Changed native preflight contract")
    expected = {r["id"]: r for r in native["records"]}
    items = load(folder)
    if len(expected) != len(items) or {i["id"] for i in items} != set(expected):
        raise ValueError("Incomplete native preflight identities")
    rows = []
    for item in items:
        before = expected[item["id"]]
        wave, rate = sf.read(folder/item["id"]/"original.wav", dtype="float32")
        if (rate != RATE or wave.ndim != 1 or len(wave)/RATE != before["duration"]
                or hashlib.sha256(wave.tobytes()).hexdigest() != before["wave_samples_sha256"]
                or item["natural_sample_expirations"] != before["natural_sample_expirations"]
                or len(item["reference"]) != before["keys"]
                or int(((item["reference"][:, 1]-item["reference"][:, 0]) > 4).sum())
                != before["holds_over_four_seconds"]):
            raise ValueError("Changed native synthesis samples, commands or expiration audit")
        target, weight = frame_supervision(item)
        if np.any(weight[target > 0]) or np.any(weight[target == 0] != 1):
            raise ValueError("Invented generated-piano presence supervision")
        selected = item["eligible"] & item["mask"]
        rows.append({"id": item["id"], "source_group": item["source_group"],
            "candidate_notes": len(item["events"]),
            "supervised_keep": int((item["y"][selected] == 1).sum()),
            "supervised_remove": int((item["y"][selected] == 0).sum()),
            "reference_keys": len(item["reference"]),
            "holds_over_four_seconds": before["holds_over_four_seconds"],
            "natural_sample_expirations": len(before["natural_sample_expirations"]),
            "unknown_positive_frames": int((weight == 0).sum()),
            "known_absent_frames": int(weight.sum()),
            "codec_lag_samples": item["codec_clock"]["codec_lag_samples"],
            "padding_adjustment_samples": item["codec_clock"]["silent_padding_adjustment_samples"],
            "native_sample_values_bit_identical": True})
    groups = Counter(i["source_group"] for i in items)
    if (len(groups) != 36 or set(groups.values()) != {3}
            or any(i["group"] != "train" for i in items)
            or any(r["codec_lag_samples"] != 0 for r in rows)):
        raise ValueError("Changed training parents or MP3 clock")
    totals = {key: sum(r[key] for r in rows) for key in (
        "candidate_notes", "supervised_keep", "supervised_remove", "reference_keys",
        "holds_over_four_seconds", "natural_sample_expirations", "unknown_positive_frames",
        "known_absent_frames")}
    result = {"version": "coco-piano-v54-strict-preparation-audit-v2",
        "recordings": len(items), "training_parent_groups": len(groups), **totals,
        "manifest_sha256": digest(folder/"manifest.json"),
        "plan_sha256": digest(folder/"plan.json"), "native_preflight_sha256": digest(preflight),
        "audit_producer_sha256": digest(Path(__file__)), "no_model_inference": True,
        "no_user_audio_or_scores": True, "no_consumed_or_fresh_tests_read": True,
        "new_weights_deployed": False, "all_codec_lags_zero": True,
        "all_native_sample_values_bit_identical": True,
        "generated_positive_support_frames_masked": True,
        "per_recording": rows}
    preserve(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_recording"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "preflight", "output"):
        parser.add_argument(name, type=Path)
    audit(**vars(parser.parse_args()))
