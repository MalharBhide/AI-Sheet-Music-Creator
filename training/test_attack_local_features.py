"""Fine attack evidence has a physical clock and does not depend on holds."""

import numpy as np
import pytest
from attack_local_features import CHANNELS, HOP, RATE, STEPS, observe, sequences


@pytest.fixture(scope='module')
def recording():
    samples=np.zeros(6*RATE,np.float32)
    for start,end in ((3.,3.4),(3.6,4.)):
        a,b=round(start*RATE),round(end*RATE)
        t=np.arange(b-a)/RATE
        samples[a:b]=(np.sin(2*np.pi*130.8128*t)+.4*np.sin(2*np.pi*261.6256*t)).astype(np.float32)
    return observe(samples,RATE)


def test_attack_clock_detects_each_repeated_note(recording):
    events=np.array([[3.,3.4,48.,70.],[3.6,4.,48.,70.]])
    frames=sequences(recording,events)
    assert frames.shape==(2,STEPS,CHANNELS)
    assert np.isfinite(frames).all() and np.all((frames>=0)&(frames<=1))
    for row in frames:
        detected=(np.argmax(row[:,17])-STEPS//2)*HOP/RATE
        assert abs(detected)<.025
        assert row[:,17].max()>.5


def test_attack_evidence_is_independent_of_note_end(recording):
    short=sequences(recording,np.array([[3.,3.1,48.,70.]]))
    held=sequences(recording,np.array([[3.,5.9,48.,70.]]))
    np.testing.assert_array_equal(short,held)


def test_late_prediction_moves_observed_attack_before_the_center(recording):
    frames=sequences(recording,np.array([[3.06,3.4,48.,70.]]))
    detected=(np.argmax(frames[0,:,17])-STEPS//2)*HOP/RATE
    assert -.085<detected<-.035


def test_silent_short_clip_preserves_its_original_duration():
    evidence=observe(np.zeros(RATE//10,np.float32),RATE)
    assert evidence['duration']==.1
    frames=sequences(evidence,np.array([[0.,.1,48.,70.]]))
    np.testing.assert_array_equal(frames,np.zeros_like(frames))
    with pytest.raises(ValueError):
        sequences(evidence,np.array([[0.,.2,48.,70.]]))


@pytest.mark.parametrize('samples,rate',[(np.array([],np.float32),RATE),
    (np.ones((3,2)),RATE),(np.array([np.nan]),RATE),(np.ones(100),44100)])
def test_invalid_waveform_is_rejected(samples,rate):
    with pytest.raises(ValueError):
        observe(samples,rate)


@pytest.mark.parametrize('event',[[3.,2.9,48.,70.],[-.1,.2,48.,70.],
    [3.,3.2,48.5,70.],[3.,3.2,120.,70.],[3.,np.inf,48.,70.]])
def test_invalid_note_clock_or_pitch_is_rejected(recording,event):
    with pytest.raises(ValueError):
        sequences(recording,np.array([event]))
