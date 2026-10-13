import hashlib
import json

import numpy as np
import prepare_keynata_piano_v57 as preparation
import pytest


def rows():
    return [{"id": name, "source_seconds": 24*windows-2, "midi": name+".mid",
             "midi_sha256": "source", "group": "train", "source_group": "keynata-generated/"+name}
            for name, windows in preparation.WINDOWS.items()]


def test_complete_fixed_parent_windows_and_dynamics():
    jobs = preparation.tasks(rows())
    assert len(jobs) == len({job["id"] for job in jobs}) == 72
    assert {job["bitrate"] for job in jobs} == {48, 128, 192}
    assert all(job["crop_start_seconds"] % 24 == 0 for job in jobs)
    for row in rows():
        assert sum(job["source_id"] == row["id"] for job in jobs) == 3*preparation.WINDOWS[row["id"]]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "duration", "unknown", "nan"])
def test_changed_fixed_parent_population_rejected(mutation):
    changed = rows()
    if mutation == "missing":
        changed.pop()
    elif mutation == "duplicate":
        changed[0] = changed[1]
    elif mutation == "unknown":
        changed[0]["id"] = "unapproved"
    else:
        changed[0]["source_seconds"] = float("nan") if mutation == "nan" else 1000
    with pytest.raises(ValueError, match="population"):
        preparation.tasks(changed)


def test_continuous_hold_is_only_pitch_support_at_a_crop_boundary():
    authored, _ = preparation.transform(np.array([[1, 30, 48, 80], [10, 11, 55, 80.]]), 0)
    # The 29-second hold crosses both 24-second crops. Neither crop gets a
    # manufactured attack/release pair for it; both retain clipped pitch support.
    for start in (0, 24):
        reference, support = preparation.labels(authored, {"crop_start_seconds": start})
        assert not np.any(reference[:, 2] == 48)
        assert np.any(support[:, 2] == 48)
    reference, _ = preparation.labels(authored, {"crop_start_seconds": 0})
    assert list(reference[:, 2]) == [55]


def receipt(tmp_path, monkeypatch):
    source = tmp_path/"source"
    source.mkdir()
    cohort = rows()
    jobs = preparation.tasks(cohort)
    events = np.array([[0, 170, 48, 80.]])
    monkeypatch.setattr(preparation, "sources", lambda root: cohort)
    monkeypatch.setattr(preparation, "notes", lambda data: events.copy())
    monkeypatch.setattr(preparation, "digest", lambda path: "digest")
    monkeypatch.setattr(preparation, "contracts", lambda: {"producer": "code"})
    monkeypatch.setattr(preparation, "piano_contracts", lambda: {"piano": "asset"})
    monkeypatch.setattr(preparation.support, "hashes", lambda: {"baseline": "asset"})
    records = []
    for row in cohort:
        (tmp_path/row["midi"]).write_bytes(b"midi")
        row["midi"] = str(tmp_path/row["midi"])
    for job in jobs:
        authored, _ = preparation.transform(events, job["variant"])
        reference, support = preparation.labels(authored, job)
        records.append({**job, "duration": 24, "sample_rate": preparation.RATE,
            "source_midi_sha256": "source", "keys": len(reference),
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "pitch_support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
            "wave_samples_sha256": "a"*64})
    native = {"version": preparation.VERSION, "code_sha256": preparation.contracts(),
        "piano_assets": preparation.piano_contracts(), "source_manifest_sha256": "digest",
        "symbolic_audit_sha256": "digest", "baseline_hashes": preparation.support.hashes(),
        "no_model_inference": True, "no_user_audio_or_scores": True, "records": records}
    path = tmp_path/"native.json"
    return source, jobs, native, path


@pytest.mark.parametrize("mutation", [None, "missing", "duplicate", "commands", "support",
    "wave", "duration", "bitrate", "source", "producer", "baseline", "audio-inference"])
def test_native_receipt_rejects_incomplete_or_changed_clock_evidence(tmp_path, monkeypatch, mutation):
    source, jobs, native, path = receipt(tmp_path, monkeypatch)
    if mutation == "missing":
        native["records"].pop()
    elif mutation == "duplicate":
        native["records"][0] = native["records"][1]
    elif mutation in ("commands", "support", "wave"):
        key = {"commands": "reference_sha256", "support": "pitch_support_sha256", "wave": "wave_samples_sha256"}[mutation]
        native["records"][0][key] = "changed"
    elif mutation in ("duration", "bitrate", "source"):
        key = {"source": "source_midi_sha256"}.get(mutation, mutation)
        native["records"][0][key] = "changed"
    elif mutation in ("producer", "baseline"):
        native[{"producer": "code_sha256", "baseline": "baseline_hashes"}[mutation]] = {}
    elif mutation == "audio-inference":
        native["no_model_inference"] = False
    path.write_text(json.dumps(native))
    if mutation:
        with pytest.raises(ValueError, match="native"):
            preparation.native_receipt(path, source, jobs)
    else:
        assert len(preparation.native_receipt(path, source, jobs)) == 72
