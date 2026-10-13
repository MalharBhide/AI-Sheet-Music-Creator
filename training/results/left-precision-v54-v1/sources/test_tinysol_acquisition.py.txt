import io
import tarfile
import wave

import pytest
from acquire_tinysol_v45 import pcm_clock, safe_member, safe_path


@pytest.mark.parametrize("name", ["../outside.wav", "/absolute.wav", "a/../escape.wav", "a\\escape.wav", ""])
def test_acquisition_refuses_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(name)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_acquisition_refuses_links_and_special_members(kind):
    member = tarfile.TarInfo("safe.wav")
    member.type = kind
    with pytest.raises(ValueError, match="type or size"):
        safe_member(member)


def test_original_pcm_clock_is_measured_and_truncation_is_rejected():
    data = io.BytesIO()
    with wave.open(data, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\x00\x00" * 44100)
    payload = data.getvalue()
    assert pcm_clock(payload)["duration"] == 1
    with pytest.raises(ValueError, match="Truncated"):
        pcm_clock(payload[:-2])


def test_member_sizes_are_bounded_before_reading():
    member = tarfile.TarInfo("large.wav")
    member.size = 16 * 1024 * 1024 + 1
    with pytest.raises(ValueError, match="type or size"):
        safe_member(member)
