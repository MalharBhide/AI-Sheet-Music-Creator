import io
import tarfile

import pytest
import yaml
from assess_cocochorales import checksum_list, read_metadata


def archive(name, data, link=False):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:bz2") as result:
        entry = tarfile.TarInfo(name)
        if link:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "outside.yaml"
            result.addfile(entry)
        else:
            entry.size = len(data)
            result.addfile(entry, io.BytesIO(data))
    return target.getvalue()


def test_metadata_retains_original_midi_grouping_without_audio_or_labels():
    data = b"midi_file: 123.mid\nensemble: string\ninstrument_name: {0: violin, 3: cello}\n"
    row = read_metadata(archive("train/string_track000001.yaml", data))[0]
    assert row["original_midi"] == "123.mid"
    assert row["instruments"][3] == "cello"
    assert len(row["member_sha256"]) == 64


@pytest.mark.parametrize(
    "name,link",
    [
        ("../escape.yaml", False),
        ("/escape.yaml", False),
        ("valid.yaml", True),
        ("audio.wav", False),
    ],
)
def test_unsafe_or_non_metadata_members_are_refused(name, link):
    with pytest.raises(ValueError):
        read_metadata(archive(name, b"anything", link))


def test_original_group_identity_and_safe_yaml_are_required():
    for data in (b"ensemble: string\n", b"!!python/object/apply:os.system ['bad']"):
        with pytest.raises((ValueError, yaml.YAMLError)):
            read_metadata(archive("metadata.yaml", data))


def test_publisher_checksum_paths_cannot_repeat_or_traverse():
    row = b"5e4419ecc5f01e5a1eebf10c70f7c7b5  ./metadata/train/1.tar.bz2\n"
    assert checksum_list(row)["metadata/train/1.tar.bz2"] == "5e4419ecc5f01e5a1eebf10c70f7c7b5"
    for data in (row + row, row.replace(b"metadata/train", b"../train"), b"not-a-checksum"):
        with pytest.raises(ValueError):
            checksum_list(data)
