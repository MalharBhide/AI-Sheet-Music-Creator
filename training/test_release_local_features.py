"""Ending evidence observes nearby held pitch without moving any candidate clocks."""

import numpy as np
import pytest
from attack_local_features import observe, sequences
from release_local_features import release_sequences


def test_release_windows_capture_supported_tail_missing_at_attack():
    rate=22050
    samples=np.zeros(4*rate,np.float32)
    start,stop=int(2.*rate),int(2.3*rate)
    samples[start:stop]=.3*np.sin(2*np.pi*(440.*2.**((48-69)/12))*np.arange(stop-start)/rate)
    events=np.array([[.8,2.,48,90]],float)
    before=events.copy()
    observed=observe(samples,rate)
    attack=sequences(observed,events)
    ending=release_sequences(observed,events)
    assert ending.shape==attack.shape==(1,61,18)
    assert ending[0,:,9:17].mean()>attack[0,:,9:17].mean()+.1
    np.testing.assert_array_equal(events,before)


def test_ending_windows_depend_on_end_but_not_start_or_velocity():
    rate=22050
    samples=np.zeros(4*rate,np.float32)
    start,stop=2*rate,round(2.3*rate)
    samples[start:stop]=np.sin(np.arange(stop-start)*.04)
    observed=observe(samples,rate)
    a=np.array([[.5,2.,48,90]],float)
    b=np.array([[1.2,2.,48,30]],float)
    np.testing.assert_array_equal(release_sequences(observed,a),release_sequences(observed,b))
    b[0,1]=2.5
    assert not np.array_equal(release_sequences(observed,a),release_sequences(observed,b))


def test_release_clock_boundary_padding_and_empty_notes_are_valid():
    rate=22050
    observed=observe(np.zeros(rate,np.float32),rate)
    ending=release_sequences(observed,np.array([[.5,1.,48,90]],float))
    assert ending.shape==(1,61,18) and np.isfinite(ending).all()
    assert np.all(ending[:,31:]==0)
    assert release_sequences(observed,np.empty((0,4))).shape==(0,61,18)


@pytest.mark.parametrize('events',[np.array([[1.,1.,48,90]]),np.array([[.5,1.01,48,90]]),
    np.array([[.5,.8,48.5,90]]),np.array([[.5,.8,48,np.nan]])])
def test_invalid_original_intervals_rejected(events):
    observed=observe(np.zeros(22050,np.float32),22050)
    with pytest.raises(ValueError):
        release_sequences(observed,events)
