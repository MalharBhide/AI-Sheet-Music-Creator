import io
import tarfile

import pytest
from audit_pdmx_archives import catalog_path, inventory, safe_path, verify_file


@pytest.mark.parametrize("name", ["/mid/a.mid", "mid/../a.mid", "other/a.mid", "mid\\a.mid", ""])
def test_unsafe_symbolic_path_rejected(name):
    with pytest.raises(ValueError):
        safe_path(name, "mid")


def test_catalog_relative_path_resolves_without_extraction():
    assert safe_path("./mid/1/6/source.mid", "mid") == "mid/1/6/source.mid"


def test_missing_catalog_sentinel_is_explicit_and_not_a_path_exception():
    assert catalog_path("NA", "mid") is None
    assert catalog_path("", "mid") is None
    assert catalog_path("./mid/1/source.mid", "mid") == "mid/1/source.mid"
    with pytest.raises(ValueError):
        catalog_path("../NA", "mid")


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as writer:
        for name, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = 3 if kind == tarfile.REGTYPE else 0
            writer.addfile(member, io.BytesIO(b"abc") if member.size else None)


@pytest.mark.parametrize("members", [
    [("mid/a.mid", tarfile.SYMTYPE)],
    [("mid/a.mid", tarfile.LNKTYPE)],
    [("mid/../a.mid", tarfile.REGTYPE)],
    [("mid/a.txt", tarfile.REGTYPE)],
    [("mid/a.mid", tarfile.REGTYPE), ("mid/a.mid", tarfile.REGTYPE)],
])
def test_archive_links_traversal_type_and_duplicates_fail_closed(tmp_path, members):
    path = tmp_path / "source.tar.gz"
    make_archive(path, members)
    with pytest.raises(ValueError):
        inventory(path, "mid", ".mid", 1)


def test_archive_inventory_has_no_extraction_side_effects(tmp_path):
    path = tmp_path / "source.tar.gz"
    make_archive(path, [("mid", tarfile.DIRTYPE), ("mid/a.mid", tarfile.REGTYPE)])
    names, counts = inventory(path, "mid", ".mid", 1)
    assert names == {"mid/a.mid"}
    assert counts["uncompressed_bytes"] == 3
    assert not (tmp_path / "mid").exists()
    with pytest.raises(ValueError, match="count"):
        inventory(path, "mid", ".mid", 2)


def test_changed_download_does_not_pass_publisher_proof(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="checksum"):
        verify_file(path, 3, "0" * 32)
    with pytest.raises(ValueError, match="checksum"):
        verify_file(path, 4, "900150983cd24fb0d6963f7d28e17f72")
