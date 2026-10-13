import hashlib
import io
import struct
import tarfile

import acquire_musicnet_later_v60 as acquisition
import pytest


def waveform(seconds=240, rate=44100):
    pattern = b"".join(struct.pack("<h", index) for index in range(256))
    size = seconds*rate*2
    raw = (pattern*((size+len(pattern)-1)//len(pattern)))[:size]
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate*2, 2, 16)
    chunks = b"fmt "+struct.pack("<I", len(fmt))+fmt+b"data"+struct.pack("<I", len(raw))+raw
    return b"RIFF"+struct.pack("<I", len(chunks)+4)+b"WAVE"+chunks, raw


def test_nonoverlapping_old_and_later_crops_preserve_original_pcm_bytes_and_clock():
    data, raw = waveform()
    starts = [15, 35, 55, 95, 155, 215]
    crops, clock = acquisition.crop_wave(io.BytesIO(data), len(data), starts)
    assert clock == {"sample_rate": 44100, "channels": 1, "format_code": 1,
                     "bits": 16, "original_duration": 240}
    for start in starts:
        assert crops[start][44:] == raw[start*44100*2:(start+20)*44100*2]
        assert len(crops[start]) == 44+20*44100*2


@pytest.mark.parametrize("starts", [[], [95, 95], [155, 95], [95, 100], [-1], [95.5], [True]])
def test_invalid_or_overlapping_source_sample_clocks_are_rejected_before_read(starts):
    with pytest.raises(ValueError, match="crop clocks"):
        acquisition.crop_wave(io.BytesIO(b""), 0, starts)


@pytest.mark.parametrize("mutation", ["truncated", "size", "rate", "short"])
def test_changed_native_container_or_incomplete_crop_is_rejected(mutation):
    data, _ = waveform(seconds=5 if mutation == "short" else 120, rate=22050 if mutation == "rate" else 44100)
    size = len(data)
    if mutation == "truncated":
        data = data[:500]
    elif mutation == "size":
        size += 1
    with pytest.raises(ValueError):
        acquisition.crop_wave(io.BytesIO(data), size, [95])


def test_reader_requires_complete_expected_object_before_verification(monkeypatch):
    data = b"verified original fixture"
    monkeypatch.setattr(acquisition, "SIZE", len(data))
    monkeypatch.setattr(acquisition, "MD5", hashlib.md5(data).hexdigest())
    monkeypatch.setattr(acquisition, "SHA256", hashlib.sha256(data).hexdigest())
    reader = acquisition.CheckedReader(io.BytesIO(data))
    reader.read(3)
    with pytest.raises(ValueError, match="Incomplete"):
        reader.verify()
    reader.read(len(data)-3)
    reader.verify()
    with pytest.raises(ValueError, match="Unbounded"):
        reader.read()


def test_reader_rejects_oversized_stream(monkeypatch):
    monkeypatch.setattr(acquisition, "SIZE", 2)
    with pytest.raises(ValueError, match="Oversized"):
        acquisition.CheckedReader(io.BytesIO(b"long")).read(4)


@pytest.mark.parametrize("name", ["/absolute", "../escape", "musicnet/../../escape", "musicnet\\escape"])
def test_unsafe_archive_paths_are_rejected(name):
    member = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="Unsafe"):
        acquisition.safe_member(member)


def test_archive_links_are_rejected():
    member = tarfile.TarInfo("musicnet/link")
    member.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="Unsafe"):
        acquisition.safe_member(member)


def test_partial_source_files_are_reused_only_when_bytes_match(tmp_path):
    path = tmp_path/"source.wav"
    acquisition.same_write(path, b"original")
    acquisition.same_write(path, b"original")
    with pytest.raises(ValueError, match="Changed partial"):
        acquisition.same_write(path, b"different")
