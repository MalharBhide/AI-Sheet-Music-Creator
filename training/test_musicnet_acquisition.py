import csv
import io
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest
from acquire_musicnet_subset import CheckedReader, crop_wave, safe_member, select


def waveform(samples, rate=44100):
    data = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def test_float_sample_clocks_preserved_exactly():
    from scipy.io import wavfile

    samples = (np.arange(75 * 44100, dtype=np.float32) / (75 * 44100)).astype(np.float32)
    wave = waveform(samples)
    crops, clock = crop_wave(io.BytesIO(wave), len(wave))
    assert clock["sample_rate"] == 44100
    for crop in crops:
        rate, actual = wavfile.read(io.BytesIO(crop["wave"]))
        assert rate == 44100
        np.testing.assert_array_equal(actual, samples[crop["start_sample"] : crop["end_sample"]])


def test_corrupt_and_short_samples_refused():
    wave = waveform(np.zeros(74 * 44100))
    with pytest.raises(ValueError, match="too short"):
        crop_wave(io.BytesIO(wave), len(wave))
    with pytest.raises(ValueError, match="member size"):
        crop_wave(io.BytesIO(wave), len(wave) + 1)
    wave = waveform(np.zeros(75 * 44100), rate=48000)
    with pytest.raises(ValueError, match="sample format"):
        crop_wave(io.BytesIO(wave), len(wave))


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("musicnet/link", tarfile.SYMTYPE),
        ("musicnet/hardlink", tarfile.LNKTYPE),
    ],
)
def test_unsafe_members_refused(name, kind):
    member = tarfile.TarInfo(name)
    member.type = kind
    with pytest.raises(ValueError, match="Unsafe"):
        safe_member(member)


def test_incomplete_stream_cannot_become_verified():
    checked = CheckedReader(io.BytesIO(b"incomplete"))
    checked.read(20)
    with pytest.raises(ValueError, match="Incomplete"):
        checked.verify()


def test_selection_work_disjoint_and_order_invariant():
    metadata = Path(__file__).parents[1] / ".training/musicnet/musicnet_metadata.csv"
    if not metadata.exists():
        pytest.skip("Downloaded original metadata is not committed")
    rows = list(csv.DictReader(metadata.open()))
    chosen = select(rows)
    assert chosen == select(list(reversed(rows)))
    assert len(chosen) == 32
    groups = {
        g: {tuple(r["work"]) for r in chosen if r["group"] == g}
        for g in ("train", "validation", "reserved")
    }
    assert not groups["train"] & (groups["validation"] | groups["reserved"])
    assert not groups["validation"] & groups["reserved"]
