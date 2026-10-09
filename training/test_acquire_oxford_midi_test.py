import io
import stat
import zipfile

import pytest
from acquire_oxford_midi_test import safe_members


def archive_with(name, payload=b"abcdef", symlink=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, payload)
    stream.seek(0)
    return stream


@pytest.mark.parametrize(
    "name", ["../escaped.wav", "/absolute.wav", "dir\\escaped.mid"]
)
def test_unsafe_member_refused(name):
    with (
        zipfile.ZipFile(archive_with(name)) as archive,
        pytest.raises(ValueError, match="Unsafe"),
    ):
        safe_members(archive)


def test_symlink_refused():
    with (
        zipfile.ZipFile(archive_with("link.wav", symlink=True)) as archive,
        pytest.raises(ValueError, match="Unsafe"),
    ):
        safe_members(archive)


def test_corrupt_member_fails_crc_before_source_proof():
    original = archive_with("piece.mid").getvalue()
    corrupt = original.replace(b"abcdef", b"xbcdef", 1)
    with (
        zipfile.ZipFile(io.BytesIO(corrupt)) as archive,
        pytest.raises(ValueError, match="Corrupt"),
    ):
        safe_members(archive)


def test_verified_inventory_keeps_original_crc_size_and_name():
    with zipfile.ZipFile(archive_with("folder/piece.mid")) as archive:
        result = safe_members(archive)
        assert result == [
            {
                "member": "folder/piece.mid",
                "bytes": 6,
                "crc32": f"{archive.infolist()[0].CRC:08x}",
            }
        ]
