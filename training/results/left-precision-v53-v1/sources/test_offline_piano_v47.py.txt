import numpy as np
import pytest
from offline_piano_v47 import BLOCK, FONT, LIBRARY, RATE, TAIL, Piano, key_clock, support_envelope


def test_authored_clocks_preserve_chords_repeats_and_long_holds():
    source = np.array([[0, 96, 40, 90], [0, 384, 44, 90], [144, 240, 40, 90]])
    events, clock = key_clock(source, 90, 12, .45)
    np.testing.assert_array_equal(events[:, 2], [52, 56, 52])
    assert events[0, 0] == events[1, 0]
    intended = 3.+source[:, :2]/96*60/90
    assert np.all(events[:, :2] >= intended)
    assert np.max(events[:, :2]-intended) < BLOCK/RATE+1e-12
    assert clock["maximum_clock_delay_seconds"] < BLOCK/RATE
    assert events[1, 1]-events[1, 0] > 2.66
    assert abs(events[2, 0]-events[0, 0]-1.) < BLOCK/RATE


def test_explicit_damping_has_no_support_outside_declared_keys_and_tails():
    events, _ = key_clock(np.array([[0, 96, 40, 90], [192, 288, 40, 90]]), 120, 0, 1)
    envelope = support_envelope(events, 6*RATE)
    assert not envelope[:round(events[0, 0]*RATE)].any()
    assert np.all(envelope[round(events[0, 0]*RATE):round(events[0, 1]*RATE)] == 1)
    assert not envelope[round((events[0, 1]+TAIL)*RATE):round(events[1, 0]*RATE)].any()
    assert not envelope[round((events[-1, 1]+TAIL)*RATE):].any()


@pytest.mark.parametrize("source,bpm,shift", [
    ([[0, 0, 40, 90]], 120, 0),
    ([[0, 96, 19, 90]], 120, 0),
    ([[0, 96, 40, 90]], float("nan"), 0),
    ([[0, 192, 40, 90], [96, 288, 40, 90]], 120, 0),
])
def test_invalid_or_overlapping_key_references_fail(source, bpm, shift):
    with pytest.raises(ValueError):
        key_clock(source, bpm, shift, 1)


@pytest.mark.skipif(not LIBRARY.is_file() or not FONT.is_file(), reason="Pinned offline renderer absent")
def test_native_sampled_piano_obeys_reference_and_explicit_silent_bounds():
    source = np.array([[0, 96, 60, 90], [0, 384, 64, 90], [192, 288, 60, 90]])
    events, _ = key_clock(source, 120, 0, 1)
    with Piano() as piano:
        wave, reference, support = piano.render(events)
        np.testing.assert_array_equal(reference, events[:, :3])
        assert wave.dtype == np.float32 and np.isfinite(wave).all()
        assert np.max(np.abs(wave)) <= .800001
        assert not wave[:round(events[0, 0]*RATE)].any()
        assert not wave[round((events[:, 1].max()+TAIL)*RATE):].any()
        assert np.max(np.abs(wave[round(4.6*RATE):round(4.9*RATE)])) > 1e-5  # Genuine long hold.
        assert support[:, 1].max() >= events[:, 1].max()+TAIL
        # Rendering a new phrase must not inherit ringing voices.
        second, _, _ = piano.render(events[:1])
        assert not second[:round(events[0, 0]*RATE)].any()
        nonzero = np.flatnonzero(second)
        assert 0 <= nonzero[0]/RATE-events[0, 0] < .02
