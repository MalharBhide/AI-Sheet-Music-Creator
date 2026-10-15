"""Byte restoration must preserve original clocks and exact crop identities."""

import copy
import io
import struct

import pytest

from restore_musicnet_bytes import BLOCK, crops, digest


def fixture():
    rate = 44100
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32) + b"\x00\x00"
    raw = b"\0" * (80 * rate * 4)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    wave = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
    crop_body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 20 * rate * 4) + raw[:20 * rate * 4]
    expected = digest(b"RIFF" + struct.pack("<I", len(crop_body) + 4) + b"WAVE" + crop_body)
    row = {"clock": {"format_code": 3, "channels": 1, "sample_rate": rate, "bits": 32,
                     "original_duration": 80.},
           "excerpts": [{"start_sample": start * rate, "end_sample": (start + 20) * rate,
                         "audio_sha256": expected} for start in (15, 35, 55)]}
    return wave, row


class Bounded(io.BytesIO):
    def read(self, size=-1):
        assert 0 <= size <= BLOCK
        return super().read(size)


def test_identical_crops_use_bounded_reads_without_consuming_unused_recording_tail():
    wave, row = fixture()
    source = Bounded(wave)
    result, count = crops(source, row)
    assert len(result) == 3 and all(digest(value) == row["excerpts"][0]["audio_sha256"] for value in result)
    assert count == source.tell() < len(wave)


@pytest.mark.parametrize("mutation", ["hash", "clock", "duration", "overlap", "coverage", "shift", "truncated", "header"])
def test_changed_source_or_reference_cannot_be_restored(mutation):
    wave, row = fixture()
    row = copy.deepcopy(row)
    if mutation == "hash":
        row["excerpts"][0]["audio_sha256"] = "0" * 64
    elif mutation == "clock":
        row["clock"]["sample_rate"] = 22050
    elif mutation == "duration":
        row["clock"]["original_duration"] = 79.
    elif mutation == "overlap":
        row["excerpts"][1]["start_sample"] = 0
    elif mutation == "coverage":
        row["excerpts"].pop()
    elif mutation == "shift":
        for crop in row["excerpts"]:
            crop["start_sample"] += 44100
            crop["end_sample"] += 44100
    elif mutation == "truncated":
        wave = wave[:1000]
    else:
        wave = b"XXXX" + wave[4:]
    with pytest.raises(ValueError):
        crops(Bounded(wave), row)
