"""Exercise release access controls without generating or reading fresh audio."""

import copy
import json
from pathlib import Path

import pytest

import fresh_approved_finetune_v70 as fresh


def protocol(tmp_path, monkeypatch):
    path = tmp_path / "training/results/approved-finetune-v70-v1/fresh-preregistration.json"
    path.parent.mkdir(parents=True)
    value = {"version": fresh.VERSION, "seeds": list(range(312101, 312141)),
             "recordings": 40, "bitrate_cycle": [48, 128, 192],
             "source_sha256": {name: name for name in fresh.FRESH_SOURCES},
             "qualification_sha256": "qualified", "no_user_audio_or_scores": True,
             "no_replacements_or_retuning": True}
    monkeypatch.setattr(fresh, "digest", lambda p: Path(p).name)
    monkeypatch.setattr(fresh, "require_qualification", lambda p: "qualified")
    path.write_text(json.dumps(value))
    return path, value


def test_fixed_independent_protocol_passes_without_audio_access(tmp_path, monkeypatch):
    protocol(tmp_path, monkeypatch)
    assert fresh.require_fresh_protocol(tmp_path) == "fresh-preregistration.json"


@pytest.mark.parametrize("field", ["version", "seeds", "recordings", "bitrate_cycle",
    "source_sha256", "qualification_sha256", "no_user_audio_or_scores", "no_replacements_or_retuning"])
def test_changed_first_pass_protocol_is_rejected(tmp_path, monkeypatch, field):
    path, value = protocol(tmp_path, monkeypatch)
    value[field] = None
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="preregistration"):
        fresh.require_fresh_protocol(tmp_path)


def proof():
    winner = {"checkpoint_sha256": "checkpoint", "threshold": .9}
    reports = {}
    for name, threshold in (("raw", .9), ("strict", .905)):
        rows = [{"id": f"dummy-{i}", "passes": True, "retained_clocks_identical": True,
                 "recovered_notes": 0, "false_notes_removed": 1,
                 "fixed_reference_identities_preserved": True,
                 "timing": {"passes": True}, "coverage": {"passes": True}} for i in range(603)]
        reports[name] = {"passes": True, "threshold": threshold,
                         "false_notes_removed": 603, "per_recording": rows}
    regression = {**reports, "passes": True, "recordings": 603,
                  "checkpoint_sha256": "checkpoint", "batch_selection_sha256": "batch-selection.json",
                  "evaluator_sha256": "evaluate_approved_finetune_v70.py",
                  "qualification_protocol_sha256": "qualified",
                  "no_test_used_for_selection": True, "no_user_audio_or_scores": True,
                  "new_weights_deployed": False}
    return regression, winner


def prerequisites(tmp_path, monkeypatch, regression, winner):
    monkeypatch.setattr(fresh, "require_fresh_protocol", lambda p: "protocol")
    monkeypatch.setattr(fresh, "require_qualification", lambda p: "qualified")
    monkeypatch.setattr(fresh, "frozen", lambda *args: (
        {"subsequent_gate": "312101-312140"}, winner, None, None))
    monkeypatch.setattr(fresh, "digest", lambda p: Path(p).name)
    (tmp_path / "consumed-regression.json").write_text(json.dumps(regression))
    def forbidden(*args, **kwargs):
        pytest.fail("Release checks must reject before generation, encoding, inference or storage mutation")
    for name in ("original", "texture", "encode", "_GeneralEngine", "storage_preflight"):
        monkeypatch.setattr(fresh, name, forbidden)


@pytest.mark.parametrize("fault", ["failed", "missing_record", "duplicate_record", "timing",
    "coverage", "identity", "qualification", "checkpoint"])
def test_incomplete_or_nonpreserving_regression_never_generates_audio(tmp_path, monkeypatch, fault):
    regression, winner = proof()
    row = regression["strict"]["per_recording"][0]
    if fault == "failed":
        regression["passes"] = False
    elif fault == "missing_record":
        regression["strict"]["per_recording"].pop()
    elif fault == "duplicate_record":
        row["id"] = "dummy-1"
    elif fault in ("timing", "coverage"):
        row[fault]["passes"] = False
    elif fault == "identity":
        row["fixed_reference_identities_preserved"] = False
    elif fault == "qualification":
        regression["qualification_protocol_sha256"] = "changed"
    elif fault == "checkpoint":
        regression["checkpoint_sha256"] = "changed"
    prerequisites(tmp_path, monkeypatch, regression, winner)
    output = tmp_path / "fresh-output"
    with pytest.raises(ValueError):
        fresh.verify(tmp_path, tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("previous", ["partial", "complete"])
def test_first_pass_cannot_be_replaced_even_after_failure(tmp_path, monkeypatch, previous):
    regression, winner = proof()
    prerequisites(tmp_path, monkeypatch, regression, winner)
    output = tmp_path / "fresh-output"
    if previous == "partial":
        output.mkdir()
        (output / "plan.json").write_text('{"interrupted": true}')
    else:
        (tmp_path / "original-first-pass.json").write_text('{"passes": false}')
    with pytest.raises(ValueError, match="Preserve"):
        fresh.verify(tmp_path, tmp_path, output)


def test_failed_validation_cannot_read_consumed_or_generate_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "require_fresh_protocol", lambda p: "protocol")
    def reject(*args):
        raise ValueError("No preserving validation winner")
    monkeypatch.setattr(fresh, "frozen", reject)
    # No regression file exists: reading it would fail this test.
    with pytest.raises(ValueError, match="validation"):
        fresh.verify(tmp_path, tmp_path, tmp_path / "fresh")


def test_changed_declared_seed_range_never_generates_replacement_clips(tmp_path, monkeypatch):
    regression, winner = proof()
    prerequisites(tmp_path, monkeypatch, regression, winner)
    monkeypatch.setattr(fresh, "frozen", lambda *args: (
        {"subsequent_gate": "303121-303140"}, copy.deepcopy(winner), None, None))
    with pytest.raises(ValueError, match="seeds"):
        fresh.verify(tmp_path, tmp_path, tmp_path / "fresh")
