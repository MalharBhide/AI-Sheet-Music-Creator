import bz2
import io
import tarfile

import pytest
from audit_cocochorales_v54 import members, piano_unambiguous, voice_events

HEADER = ",volume,vol_fluc,vibrato,brightness,attack,vol_peak_pos,pitch,onset,offset,note_length\n"
ZERO = "0,0,0,0,0,0,0,0,0,0,0\n"
NOTE = "1,.7,.2,.3,.4,.5,.6,48,0,1500,1500\n"


def test_authored_four_millisecond_clock_preserves_long_hold():
    assert voice_events((HEADER + ZERO + NOTE).encode()) == [(0, 1500, 48)]
    assert 1500 * .004 == 6.


@pytest.mark.parametrize("note", [
    NOTE.replace(",1500,1500", ",1500,1499"), NOTE.replace(",48,0,", ",48.5,0,"),
    NOTE.replace(",1500,1500", ",0,0"), NOTE.replace(".7", "nan"),
    NOTE.replace(".7", "1.1"), NOTE.replace(",48,0,", ",0,0,"),
])
def test_rejects_unknown_or_inconsistent_clock_labels(note):
    with pytest.raises(ValueError):
        voice_events((HEADER + ZERO + note).encode())


def test_rejects_overlapping_rows_in_one_voice():
    repeated = "2,.7,.2,.3,.4,.5,.6,50,1499,1700,201\n"
    with pytest.raises(ValueError):
        voice_events((HEADER + ZERO + NOTE + repeated).encode())


def test_cross_voice_unison_overlap_is_not_guessed_into_piano_releases():
    assert not piano_unambiguous([(0, 100, 48), (50, 150, 48)])
    assert piano_unambiguous([(0, 100, 48), (100, 150, 48), (0, 150, 55)])


def archive(path, names, *, symlink=False):
    with tarfile.open(path, "w:bz2") as output:
        for name in names:
            item = tarfile.TarInfo(name)
            if symlink:
                item.type, item.linkname = tarfile.SYMTYPE, "elsewhere"
                output.addfile(item)
            else:
                item.size = 1
                output.addfile(item, io.BytesIO(b"x"))


@pytest.mark.parametrize("names", [
    ["../string_track000001/0_violin.csv"], ["/string_track000001/0_violin.csv"],
    ["string_track000001/0_violin.csv"] * 2, ["string_track192001/0_violin.csv"],
    ["string_track000001/0_cello.csv"], ["string_track000001/3_cello.pickle"],
])
def test_inventory_rejects_unsafe_paths_partitions_and_types(tmp_path, names):
    path = tmp_path / "source.tar.bz2"
    archive(path, names)
    with pytest.raises(ValueError):
        list(members(path, "expression"))


def test_inventory_never_follows_links_or_extracts_members(tmp_path):
    path = tmp_path / "source.tar.bz2"
    archive(path, ["string_track000001/0_violin.csv"], symlink=True)
    with pytest.raises(ValueError):
        list(members(path, "expression"))
    assert list(tmp_path.iterdir()) == [path]


def test_publisher_parallel_bzip_concatenated_streams_are_read_completely(tmp_path):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as output:
        item = tarfile.TarInfo("string_track000001/0_violin.csv")
        item.size = 1
        output.addfile(item, io.BytesIO(b"x"))
    data = raw.getvalue()
    path = tmp_path / "source.tar.bz2"
    path.write_bytes(bz2.compress(data[:512]) + bz2.compress(data[512:]))
    assert list(members(path, "expression")) == [(1, 0, b"x")]
