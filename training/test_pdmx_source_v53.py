import copy
import json

import pytest
from acquire_pdmx_piano_v53 import URL, piano_metadata
from prepare_pdmx_piano_v53 import tasks


def metadata():
    row = {"composer_name": "Domenico Scarlatti", "license": "cc-zero", "license_url": URL,
           "subset:no_license_conflict": "True", "subset:all_valid": "True", "tracks": "0"}
    data = {"score": {"composer_name": "Domenico Scarlatti", "license": "cc-zero", "parts": 1,
        "instruments": [{"name": "Piano", "url_to_search": "/sheetmusic/piano", "count": 1}]},
        "license_url": URL}
    return row, {"data": data}


def test_explicit_matching_grant_and_solo_piano():
    row, data = metadata()
    assert piano_metadata(row, data)


@pytest.mark.parametrize("key,value", [
    ("composer_name", "Unknown"), ("license", "cc-by-nc"), ("license_url", ""),
    ("subset:no_license_conflict", "False"), ("subset:all_valid", "False"), ("tracks", "0-0"),
])
def test_changed_catalog_rights_and_identity_rejected(key, value):
    row, data = metadata()
    row[key] = value
    assert not piano_metadata(row, data)


@pytest.mark.parametrize("key,value", [
    ("composer_name", "Unknown"), ("license", "cc-by-nc"), ("parts", 2),
    ("instruments", [{"name": "Harpsichord", "count": 1}]), ("is_blocked", True),
    ("publisher", "Modern publisher"),
])
def test_changed_metadata_rights_or_intent_rejected(key, value):
    row, data = metadata()
    changed = copy.deepcopy(data)
    changed["data"]["score"][key] = value
    assert not piano_metadata(row, changed)


def test_disputed_metadata_rejected():
    row, data = metadata()
    data["data"]["opened_dispute"] = True
    assert not piano_metadata(row, data)


def test_all_windows_and_dynamics_stay_in_whole_train_source():
    rows = [{"id": "a", "source_seconds": 181.55873490416332},
            {"id": "b", "source_seconds": 198.9479166666655}]
    jobs = tasks(rows)
    assert len(jobs) == len({j["id"] for j in jobs}) == 45
    assert {j["source_id"] for j in jobs} == {"a", "b"}
    assert all(j["crop_start_seconds"] % 24 == 0 for j in jobs)
    assert {j["bitrate"] for j in jobs} == {48, 128, 192}


def test_loader_accepts_scalar_dynamics_and_rejects_changed_variants(tmp_path, monkeypatch):
    import prepare_pdmx_piano_v53 as preparation

    rows = [{"id": "a", "source_seconds": 181.55873490416332},
            {"id": "b", "source_seconds": 198.9479166666655}]
    jobs = tasks(rows)
    records = [{**j, "group": "train"} for j in jobs]
    source = tmp_path / "source"
    plan = {"version": preparation.VERSION, "source_root": str(source),
        "baseline_hashes": {}, "code_sha256": {}, "piano_assets": {}, "sample_rate": preparation.RATE,
        "source_manifest_sha256": "source", "variants": [.45, .7, 1.], "tasks": jobs,
        "expected_counts": {"train": 45}, "human_timing_labels": False,
        "no_reserved_or_consumed_audio_inference": True, "no_user_audio_or_scores": True}
    manifest = {"version": preparation.VERSION, "items": records, "plan_sha256": "plan",
        "human_timing_labels": False, "no_reserved_or_consumed_audio_inference": True,
        "no_user_audio_or_scores": True}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(preparation, "sources", lambda p: rows)
    monkeypatch.setattr(preparation.support, "hashes", lambda: {})
    monkeypatch.setattr(preparation, "contracts", lambda: {})
    monkeypatch.setattr(preparation, "piano_contracts", lambda: {})
    monkeypatch.setattr(preparation, "digest", lambda p: "source" if p == source / "manifest.json"
        else preparation.ASSETS[preparation.LICENSE] if p.name == "PIANO-SAMPLE-LICENSE.txt" else "plan")
    monkeypatch.setattr(preparation, "observation", lambda folder, record, *args: record["id"])
    assert preparation.load(tmp_path) == [j["id"] for j in jobs]
    plan["variants"][0] = .5
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Changed"):
        preparation.load(tmp_path)
