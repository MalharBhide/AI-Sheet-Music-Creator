import copy
import json

import pytest
from ensemble_contract_v56 import frozen, load_member, validation_qualifies


def report():
    timed = []
    for index in range(381):
        corpus = "musicnet-real-piano" if index < 15 else "nrgcp-authored-sampled-piano" if index < 51 else "other"
        timed.append({"id": f"timed-{index}", "corpus": corpus,
                      "false_notes_removed": int(index in (0, 15, 51)), "passes": True,
                      "fixed_reference_identities_preserved": True, "retained_clocks_identical": True,
                      "recovered_notes": 0, "timing": {"passes": True}})
    pitch = [{"id": f"pitch-{i}", "corpus": "tinysol-isolated-pitch", "passes": True,
              "wrong_pitch_gain": int(i == 0), "retained_clocks_identical": True,
              "recovered_notes": 0, "removed_fundamental_candidates": 0,
              "timing_supervision_available": False} for i in range(237)]
    result = {}
    for n in ("raw", "strict"):
        result[n] = {"per_recording": copy.deepcopy(timed), "threshold": .9, "passes": True,
                     "false_notes_removed": 3, "real_piano_gain": 1, "nrgcp_gain": 1}
        result[n+"_pitch"] = {"per_recording": copy.deepcopy(pitch), "threshold": .9, "passes": True,
                              "wrong_pitch_gain": 1, "removed_fundamental_candidates": 0,
                              "timing_supervision_available": False}
    return result


def test_complete_joint_proof_requires_preservation_and_every_separate_gain():
    assert validation_qualifies(report())
    for policy in ("raw", "strict"):
        for metric in ("real_piano_gain", "false_notes_removed", "nrgcp_gain"):
            value = report()
            value[policy][metric] = 0
            assert not validation_qualifies(value)
        value = report()
        value[policy+"_pitch"]["wrong_pitch_gain"] = 0
        assert not validation_qualifies(value)


@pytest.mark.parametrize("kind", ["missing_record", "duplicate", "changed_identity", "timing", "restore", "fabricated_gain", "pitch_identity", "invented_timing", "different_cohort"])
def test_aggregate_positive_scores_cannot_hide_missing_or_regressed_recordings(kind):
    value = report()
    if kind == "missing_record":
        value["strict"]["per_recording"].pop()
    elif kind == "duplicate":
        value["strict"]["per_recording"][1]["id"] = "timed-0"
    elif kind == "changed_identity":
        value["raw"]["per_recording"][0]["fixed_reference_identities_preserved"] = False
    elif kind == "timing":
        value["strict"]["per_recording"][0]["timing"]["passes"] = False
    elif kind == "restore":
        value["raw"]["per_recording"][0]["recovered_notes"] = 1
    elif kind == "fabricated_gain":
        value["raw"]["real_piano_gain"] = 10
    elif kind == "pitch_identity":
        value["strict_pitch"]["per_recording"][0]["removed_fundamental_candidates"] = 1
    elif kind == "invented_timing":
        value["strict_pitch"]["per_recording"][0]["timing_supervision_available"] = True
    else:
        value["strict_pitch"]["per_recording"][0]["id"] = "replacement"
    assert not validation_qualifies(value)


def test_nonqualifying_joint_run_is_rejected_before_any_checkpoint_or_test_access(tmp_path, monkeypatch):
    import ensemble_contract_v56

    (tmp_path/"plan.json").write_text("{}")
    (tmp_path/"batch-selection.json").write_text(json.dumps({"selected": False}))
    monkeypatch.setattr(ensemble_contract_v56, "protocol", lambda _: ({}, {}))
    with pytest.raises(ValueError, match="qualifying"):
        frozen(tmp_path, tmp_path)
    # No source checkpoint, validation file or test folder was present/read.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["batch-selection.json", "plan.json"]


def test_changed_checkpoint_fails_before_deserialization(tmp_path):
    (tmp_path/"changed.pt").write_bytes(b"not an executable checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        load_member({"run": str(tmp_path), "checkpoint": "changed.pt", "checkpoint_sha256": "0"*64})
