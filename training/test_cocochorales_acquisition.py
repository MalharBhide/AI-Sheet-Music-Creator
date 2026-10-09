import bz2
import hashlib
import io
import tarfile

import pytest
from acquire_cocochorales_subset import CheckedReader, consume, safe_path, select


def source_archive(entries):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return bz2.compress(result.getvalue())


def checked(data, expected=None):
    metadata = {"name": "fixture", "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}
    return CheckedReader(io.BytesIO(data), expected or metadata)


def test_complete_stream_and_selected_members_only(tmp_path):
    data = source_archive(
        [("./wanted/mix.wav", b"source samples"), ("other/mix.wav", b"not retained")]
    )
    reader = checked(data)
    rows = consume(reader, {"wanted": {}}, "main_dataset", tmp_path)
    assert set(rows) == {"wanted/mix.wav"}
    assert (tmp_path / "main_dataset/wanted/mix.wav").read_bytes() == b"source samples"
    assert not (tmp_path / "main_dataset/other").exists()
    assert reader.count == len(data)


def test_truncation_corrupt_footer_and_wrong_hash_are_refused(tmp_path):
    data = source_archive([("wanted/0_cello.csv", b"timestamps")])
    for source in (data[:-10], data[:-1] + bytes([data[-1] ^ 0xFF])):
        with pytest.raises((EOFError, OSError, ValueError)):
            consume(checked(source), {"wanted": {}}, "note_expression", tmp_path)
    with pytest.raises(ValueError, match="corrupt"):
        consume(
            checked(data, {"name": "fixture", "bytes": len(data), "md5": "0" * 32}),
            {"wanted": {}},
            "note_expression",
            tmp_path,
        )


def test_duplicate_normalized_names_and_unknown_selected_files_are_refused(tmp_path):
    data = source_archive([("./wanted/mix.wav", b"a"), ("wanted/mix.wav", b"b")])
    with pytest.raises(ValueError, match="Duplicate"):
        consume(checked(data), {"wanted": {}}, "main_dataset", tmp_path)
    data = source_archive([("wanted/parameters.pickle", b"untrusted")])
    with pytest.raises(ValueError, match="Unexpected selected"):
        consume(checked(data), {"wanted": {}}, "main_dataset", tmp_path)


@pytest.mark.parametrize("name", ["../outside.wav", "/outside.wav", "wanted\\outside.wav"])
def test_unsafe_paths_are_refused_without_extraction(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tarfile.TarInfo(name))


def test_source_selection_is_disjoint_and_metadata_only():
    archives = []
    for first in (0, 2000):
        rows = [
            {
                "member": f"string_track{first + i:06}.yaml",
                "original_midi": f"{first + i}.mid",
                "ensemble": "string",
            }
            for i in range(2000)
        ]
        archives.append({"records": rows})
    chosen = select({"archives": archives})
    assert len(chosen) == len({r["original_midi"] for r in chosen}) == 18
    assert [sum(r["group"] == group for r in chosen) for group in ("train", "validation")] == [
        12,
        6,
    ]
    archives[1]["records"][-1]["original_midi"] = archives[0]["records"][-1]["original_midi"]
    with pytest.raises(ValueError, match="leakage"):
        select({"archives": archives})


def test_checked_reader_refuses_unbounded_requests():
    with pytest.raises(ValueError, match="Unbounded"):
        checked(b"data").read()
