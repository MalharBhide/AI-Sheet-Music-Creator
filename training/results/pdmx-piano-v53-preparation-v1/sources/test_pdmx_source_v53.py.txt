import copy

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
