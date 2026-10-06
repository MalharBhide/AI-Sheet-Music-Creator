import io
import zipfile

import pytest
from acquire_groove_subset import MAX_MEMBER, RangeReader, safe_member, select


def test_performer_split_uses_metadata_only_and_excludes_original_tests():
    rows = []
    for index in range(1, 11):
        for j in range(3):
            rows.append(
                {
                    "drummer": f"drummer{index}",
                    "audio_filename": f"{index}-{j}.wav",
                    "midi_filename": f"{index}-{j}.mid",
                    "id": f"{index}-{j}",
                    "style": ("jazz" if j != 1 else "rock") + "/beat",
                    "split": "test" if index == 2 or j == 2 else "train",
                    "duration": "19",
                }
            )
    records = select(rows)
    assert len(records) == 18
    assert all(r["split"] != "test" for r in records)
    assert not any(r["drummer"] == "drummer2" for r in records)
    groups = {
        g: {r["drummer"] for r in records if r["group"] == g}
        for g in ("train", "validation", "reserved")
    }
    assert list(map(len, groups.values())) == [5, 2, 2]
    assert not (
        groups["train"] & groups["validation"]
        | groups["train"] & groups["reserved"]
        | groups["validation"] & groups["reserved"]
    )


@pytest.mark.parametrize("name", ["../escape.wav", "/escape.wav", "groove\\escape.wav"])
def test_zip_traversal_refused(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_member(zipfile.ZipInfo(name))


def test_zip_symlinks_oversized_members_and_missing_verification_refused():
    member = zipfile.ZipInfo("groove/drum.wav")
    member.external_attr = 0o120777 << 16
    with pytest.raises(ValueError, match="Unsafe"):
        safe_member(member)
    member.external_attr = 0
    member.file_size = MAX_MEMBER + 1
    with pytest.raises(ValueError, match="oversized"):
        safe_member(member)
    with pytest.raises(ValueError, match="complete publisher"):
        RangeReader({"sha256": "unverified", "bytes": 0})


def test_changed_checksum_stops_before_creating_output(tmp_path, monkeypatch):
    import acquire_groove_subset as module

    def reject(_):
        raise ValueError("Corrupt publisher archive")

    monkeypatch.setattr(module, "verified_source", reject)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Corrupt"):
        module.acquire(tmp_path / "witness.json", output)
    assert not output.exists()


def test_changed_range_identity_stops_before_reading_payload(monkeypatch):
    import acquire_groove_subset as module

    class WrongObject(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-3/changed"}

        def read(self, *_):
            raise AssertionError("Unverified range payload must not be read")

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: WrongObject())
    reader = RangeReader(
        {
            "sha256": module.SHA,
            "bytes": module.SIZE,
            "complete_original_object_verified": True,
            "generation": "123",
            "etag": '"abcd"',
        }
    )
    with pytest.raises(ValueError, match="immutable Groove range"):
        reader.read(4)
