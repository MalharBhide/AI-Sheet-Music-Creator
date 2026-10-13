import io
import tarfile

import mido
import pytest
from acquire_nrgcp_v47 import family, member_path, notes, partition, pattern


def midi_bytes(messages):
    midi = mido.MidiFile(type=1, ticks_per_beat=96)
    track = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("time_signature", numerator=4, denominator=4),
        mido.Message("program_change", program=0), *messages])
    used = sum(m.time for m in track)
    track.append(mido.MetaMessage("end_of_track", time=3072-used))
    midi.tracks.append(track)
    stream = io.BytesIO()
    midi.save(file=stream)
    return stream.getvalue()


def test_source_key_duration_comes_from_unambiguous_on_and_off():
    data = midi_bytes([mido.Message("note_on", note=40, velocity=90),
        mido.Message("note_on", note=40, velocity=0, time=144),
        mido.Message("note_on", note=40, velocity=91, time=48),
        mido.Message("note_on", note=40, velocity=0, time=96)])
    assert notes(data) == [(0, 144, 40, 90), (192, 288, 40, 91)]


def test_overlapping_same_key_is_excluded_without_guessed_duration_repair():
    data = midi_bytes([mido.Message("note_on", note=40, velocity=90),
        mido.Message("note_on", note=40, velocity=91, time=48),
        mido.Message("note_on", note=40, velocity=0, time=48),
        mido.Message("note_on", note=40, velocity=0, time=48)])
    assert notes(data) is None


@pytest.mark.parametrize("messages", [
    [mido.Message("note_on", note=40, velocity=90)],
    [mido.Message("note_on", note=40, velocity=0)],
    [mido.Message("note_on", note=40, velocity=90), mido.Message("note_on", note=40, velocity=0)],
    [mido.Message("control_change", control=64, value=127)],
    [mido.Message("note_on", channel=9, note=40, velocity=90)],
])
def test_missing_or_ambiguous_clocks_and_pedal_are_not_invented(messages):
    with pytest.raises(ValueError):
        notes(midi_bytes(messages))


def test_related_harmonic_and_exact_transposed_variants_share_groups():
    first = [(0, 96, 40, 90), (0, 96, 44, 90), (0, 96, 47, 90), (384, 480, 45, 90)]
    transposed = [(a, b, p+7, 95) for a, b, p, v in first]
    rhythmic = [(a+48, b+96, p+12, 90) for a, b, p, v in first]
    assert family(first) == family(transposed) == family(rhythmic)
    assert pattern(first) == pattern(transposed)
    assert partition(family(first)) == partition(family(transposed))


@pytest.mark.parametrize("name,kind", [
    ("../escape_nrgcp_dataset.mid", tarfile.REGTYPE),
    ("/escape_nrgcp_dataset.mid", tarfile.REGTYPE),
    ("sub/file_nrgcp_dataset.mid", tarfile.REGTYPE),
    ("safe_nrgcp_dataset.mid", tarfile.SYMTYPE),
    ("safe_nrgcp_dataset.mid", tarfile.LNKTYPE),
])
def test_source_archive_paths_and_links_are_bounded(name, kind):
    item = tarfile.TarInfo(name)
    item.type, item.size = kind, 200
    with pytest.raises(ValueError, match="member"):
        member_path(item)


def test_large_or_non_midi_archive_member_is_rejected():
    item = tarfile.TarInfo("safe_nrgcp_dataset.mid")
    item.size = 16385
    with pytest.raises(ValueError, match="member"):
        member_path(item)
