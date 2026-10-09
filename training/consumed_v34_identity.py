"""Bind all 519 consumed identities without reading regression features or outcomes."""

import json

from prepare_robust_training_stems import digest


def declaration(previous, first_pass):
    paths = [previous / "manifest.json", first_pass / "manifest.json"]
    manifests = [json.loads(p.read_text()) for p in paths]
    old, new = [m["items"] for m in manifests]
    if (
        len(old) != 469
        or len(new) != 50
        or any(r["group"] != "test" for r in old + new)
        or not manifests[1]["now_consumed_regression"]
        or not manifests[1]["no_fitting_selection_user_audio_or_scores"]
        or len({r["id"] for r in old + new}) != 519
        or manifests[1]["plan_sha256"] != digest(first_pass / "plan.json")
    ):
        raise ValueError("Incomplete consumed V34 identity declaration")
    return {
        "recordings": 519,
        "roots": [str(previous), str(first_pass)],
        "manifest_sha256": [digest(p) for p in paths],
        "plan_sha256": [digest(root / "plan.json") for root in (previous, first_pass)],
        "ids": [r["id"] for r in old + new],
        "metadata_only_no_features_or_metrics": True,
    }


def first_pass_declaration(oxford):
    from oxford_midi_test_source import declaration as verified_pairs

    pairs = verified_pairs(oxford)
    return {
        "oxford": pairs,
        "policy": "All eight original piano/MIDI pairs as one reserved pianist group. First 30 seconds, or the complete source if shorter, from source time zero. Original publisher alignment only, no prediction-based timestamp adjustment. Held-at-crop notes protect pitch coverage but are not new attacks. Refuse malformed clocks or unusable publisher alignment; do not replace clips.",
        "procedural_seeds": list(range(280101, 280133)),
        "recordings": 40,
        "separate_real_piano_positive_gain_both_settings": True,
        "no_audio_or_label_interpretation_before_frozen_regression_pass": True,
        "no_retuning_runner_up_or_replacement_first_pass": True,
    }
