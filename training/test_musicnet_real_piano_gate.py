import copy

import pytest
from gate_musicnet_real_piano_v33 import real_benefit


def fixture():
    ids = [f"piano-{i}" for i in range(18)]
    rows = [
        {
            "id": name,
            "corpus": "reserved-musicnet-real-piano",
            "passes": True,
            "baseline": {"true_positives": 5, "false_positives": 2},
            "candidate": {"true_positives": 5, "false_positives": 2},
            "coverage": {"passes": True},
            "timing": {"passes": True},
            "retimed_notes": 0,
            "matches": {"attack": {"lost": 0}, "hold": {"lost": 0}},
        }
        for name in ids
    ]
    return {
        name: {"per_recording": copy.deepcopy(rows)} for name in ("raw", "margin")
    }, {"jobs": [{"id": name} for name in ids]}


def test_authored_only_benefit_cannot_qualify_real_piano():
    report, declaration = fixture()
    for setting in report.values():
        setting["false_notes_removed"] = 99
        setting["per_recording"].append({"id": "authored", "corpus": "authored"})
    with pytest.raises(ValueError, match="No independently positive"):
        real_benefit(report, declaration)


def test_both_settings_need_positive_gain_and_every_real_recording_preserved():
    report, declaration = fixture()
    report["raw"]["per_recording"][0]["candidate"]["false_positives"] = 1
    with pytest.raises(ValueError, match="No independently positive"):
        real_benefit(report, declaration)
    report["margin"]["per_recording"][0]["candidate"]["false_positives"] = 1
    assert real_benefit(report, declaration) == {
        name: {"false_notes_removed": 1, "recordings": 18} for name in ("raw", "margin")
    }
    report["margin"]["per_recording"][-1]["coverage"]["passes"] = False
    with pytest.raises(ValueError, match="preservation"):
        real_benefit(report, declaration)


def test_missing_real_piano_recording_is_refused():
    report, declaration = fixture()
    report["raw"]["per_recording"].pop()
    with pytest.raises(ValueError, match="identities"):
        real_benefit(report, declaration)
