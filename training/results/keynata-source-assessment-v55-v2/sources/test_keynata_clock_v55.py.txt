import io

import mido
import numpy as np
import pytest
from keynata_piano_clock_v55 import notes
from pdmx_piano_clock_v53 import notes as strict_notes


def midi(messages):
    source = mido.MidiFile(ticks_per_beat=480)
    source.tracks.append(mido.MidiTrack(messages))
    output = io.BytesIO()
    source.save(file=output)
    return output.getvalue()


def keys():
    return [mido.Message("note_on", note=48, velocity=80),
            mido.Message("note_off", note=48, time=960)]


def test_redundant_neutral_initial_resets_leave_key_clock_bit_identical():
    original = keys()
    with_reset = [mido.Message("pitchwheel", pitch=0), *original[:1],
                  mido.Message("program_change", program=0), *original[1:]]
    np.testing.assert_array_equal(notes(midi(with_reset)), strict_notes(midi(original)))


@pytest.mark.parametrize("reset", [mido.Message("pitchwheel", pitch=1),
    mido.Message("pitchwheel", pitch=0, time=1), mido.Message("pitchwheel", pitch=0, channel=9),
    mido.Message("program_change", program=1), mido.Message("program_change", program=0, channel=9)])
def test_actual_pitch_program_or_late_reset_is_not_erased(reset):
    with pytest.raises(ValueError):
        notes(midi([reset, *keys()]))


def test_nonzero_tick_neutral_pitch_reset_is_never_silently_discarded():
    with pytest.raises(ValueError, match="late"):
        notes(midi([*keys(), mido.Message("pitchwheel", pitch=0)]))


def test_same_key_overlap_still_rejected():
    with pytest.raises(ValueError, match="overlap"):
        notes(midi([keys()[0], keys()[0], keys()[1]]))


def test_pedal_still_rejected():
    with pytest.raises(ValueError, match="controller"):
        notes(midi([mido.Message("control_change", control=64, value=127), *keys()]))
