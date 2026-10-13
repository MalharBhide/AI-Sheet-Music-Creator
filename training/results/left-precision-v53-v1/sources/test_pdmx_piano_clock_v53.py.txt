import io

import mido
import numpy as np
import pytest
from pdmx_piano_clock_v53 import crop_labels, excerpt, notes, synthesis_clock


def midi(messages, tracks=None):
    score = mido.MidiFile(ticks_per_beat=480)
    score.tracks.extend(tracks or [mido.MidiTrack(messages)])
    stream = io.BytesIO()
    score.save(file=stream)
    return stream.getvalue()


def on(note=48, time=0, **kw):
    return mido.Message("note_on", note=note, velocity=80, time=time, **kw)


def off(note=48, time=480):
    return mido.Message("note_off", note=note, time=time)


def test_tempo_change_during_hold_and_zero_velocity_release():
    events = notes(midi([on(), mido.MetaMessage("set_tempo", tempo=1000000, time=480),
                         mido.Message("note_on", note=48, velocity=0, time=480)]))
    np.testing.assert_array_equal(events, [[0, 1.5, 48, 80]])


def test_simultaneous_chord_and_repeated_attack():
    events = notes(midi([on(), on(55), off(), off(55, 0), on(), off()]))
    np.testing.assert_array_equal(events, [[0, .5, 48, 80], [0, .5, 55, 80], [.5, 1, 48, 80]])


@pytest.mark.parametrize("messages", [
    [on(), on(time=20), off()], [off()], [on()], [on(), off(time=0)],
    [on(), on(55, channel=1), off()], [on(channel=9), off()],
    [mido.Message("control_change", control=64, value=127), on(), off()],
    [on(), mido.Message("control_change", control=7, value=100), off()],
    [mido.Message("program_change", program=1), on(), off()],
    [mido.Message("pitchwheel"), on(), off()], [on(20), off(20)],
    [mido.MetaMessage("set_tempo", tempo=500000),
     mido.MetaMessage("set_tempo", tempo=600000), on(), off()],
])
def test_rejects_ambiguous_or_unsupported_labels(messages):
    with pytest.raises(ValueError):
        notes(midi(messages))


def test_cross_track_conflicting_tempo_rejected():
    tracks = [mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=t)]) for t in (500000, 600000)]
    tracks.append(mido.MidiTrack([on(), off()]))
    with pytest.raises(ValueError, match="Conflicting"):
        notes(midi([], tracks))


def test_initial_default_controllers_allowed():
    control = mido.Message("control_change", control=121, value=0)
    np.testing.assert_array_equal(notes(midi([control, on(), off()])), [[0, .5, 48, 80]])


def test_excerpt_refuses_crossing_holds_and_tail():
    events = np.array([[1, 9, 48, 80], [10, 11, 55, 80]], float)
    assert excerpt(events, 3) is None
    assert excerpt(events, 9.1) is None
    np.testing.assert_allclose(excerpt(events, 9.3), [[3.7, 4.7, 55, 80]], rtol=0, atol=1e-14)
    assert excerpt(events, 0, 8) is None


def test_rounding_preserves_chord_and_recurrence_without_splitting_hold():
    events = np.array([[3, 5.231, 48, 80], [3, 4, 55, 60], [5.231, 6, 48, 70]])
    result = synthesis_clock(events, .5)
    assert result[0, 0] == result[1, 0]
    assert result[0, 1] == result[2, 0]
    assert result[0, 1] - result[0, 0] > 2.23
    np.testing.assert_array_equal(result[:, 3], [40, 30, 35])


def test_crop_keeps_crossing_pitch_but_does_not_invent_attacks_or_releases():
    reference = np.array([[1, 30, 48], [10, 11, 55], [25, 29, 60]], float)
    support = reference.copy()
    support[:, 1] += .4
    attacks, present = crop_labels(reference, support, 3, 24)
    np.testing.assert_array_equal(attacks, [[7, 8, 55]])
    np.testing.assert_allclose(present, [[0, 24, 48], [7, 8.4, 55], [22, 24, 60]])
