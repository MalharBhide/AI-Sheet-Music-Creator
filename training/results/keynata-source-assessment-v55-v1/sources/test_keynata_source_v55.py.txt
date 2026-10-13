import hashlib

import pytest
from acquire_keynata_piano_v55 import verify


def entry(data):
    return {"type": "blob", "mode": "100644", "size": len(data),
            "sha": hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data,
                                usedforsecurity=False).hexdigest()}


def test_publisher_git_blob_authentication_and_local_sha256():
    data = b"declared source bytes"
    assert verify(data, entry(data)) == hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="Changed"):
        verify(data+b"changed", entry(data))


@pytest.mark.parametrize("key,value", [("type", "tree"), ("mode", "120000"),
                                      ("size", 999), ("sha", "0"*40)])
def test_non_blob_link_or_changed_source_rejected(key, value):
    data = b"source"
    original = entry(data)
    original[key] = value
    with pytest.raises(ValueError):
        verify(data, original)


@pytest.mark.parametrize("mutation", [None, "license", "id", "instrument", "asset", "screening"])
def test_individual_metadata_must_authenticate_piano_identity(mutation):
    from audit_keynata_piano_v55 import metadata_matches

    row = {k: 0 for k in ("bpm", "duration_seconds", "engine_commit", "form", "generated_date",
        "length_tier", "meter_beats", "mode", "modulation_semitones", "section_length", "seed", "tonic")}
    row.update(track_id="classical_piano_4001", category="classical_piano",
               instrumentation=["piano"], versions=["main"])
    asset = {"format": "midi", "name": "classical_piano_4001.mid", "role": "main"}
    metadata = {**row, "files": [asset], "screening": {"passed": True},
                "license": {"generated_content": "CC0-1.0"}}
    if mutation == "license":
        metadata["license"]["generated_content"] = "unknown"
    elif mutation == "id":
        metadata["track_id"] = "other"
    elif mutation == "instrument":
        metadata["instrumentation"] = ["violin"]
    elif mutation == "asset":
        asset["name"] = "../other.mid"
    elif mutation == "screening":
        metadata["screening"]["passed"] = False
    if mutation:
        with pytest.raises(ValueError):
            metadata_matches(row, metadata)
    else:
        assert metadata_matches(row, metadata) is asset
