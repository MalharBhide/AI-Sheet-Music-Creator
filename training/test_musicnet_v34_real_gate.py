"""A validation winner cannot ship on authored-only first-pass improvements."""

import copy

import pytest
from gate_musicnet_real_piano_v34 import real_benefit
from test_musicnet_real_piano_gate import fixture


def test_v34_rejects_authored_gain_without_real_piano_benefit():
    report, declaration = fixture()
    for setting in report.values():
        setting["false_notes_removed"] = 1
    with pytest.raises(ValueError, match="No independently positive"):
        real_benefit(report, declaration)


def test_v34_requires_real_benefit_at_both_frozen_settings():
    report, declaration = fixture()
    report["raw"]["per_recording"][0]["candidate"]["false_positives"] -= 1
    with pytest.raises(ValueError, match="No independently positive"):
        real_benefit(report, declaration)
    report["margin"] = copy.deepcopy(report["raw"])
    assert all(v["false_notes_removed"] == 1 for v in real_benefit(report, declaration).values())
    report["margin"]["per_recording"][-1]["matches"]["hold"]["lost"] = 1
    with pytest.raises(ValueError, match="preservation"):
        real_benefit(report, declaration)


def test_v34_refuses_missing_real_recordings():
    report, declaration = fixture()
    report["raw"]["per_recording"].pop()
    with pytest.raises(ValueError, match="identities"):
        real_benefit(report, declaration)
