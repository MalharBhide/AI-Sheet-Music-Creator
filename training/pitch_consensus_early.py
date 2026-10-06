"""Early-only uniform attack corrections cannot shorten retained intervals."""

import numpy as np
from bass_joint_timing import EARLY, KEEP, LATE, REMOVE, protected
from bass_temporal_data import annotate
from pitch_consensus_timing import transform as prior_transform
from score_bass_joint_timing import timing

VERSION='bass-v16-early-consensus-attack-v22-v1'


def transform(events, action, duration, strength=1.):
    # A later onset at a fixed release shortens pitch coverage. This experiment
    # makes that action unavailable in both supervision and runtime decisions.
    return prior_transform(events,np.where(np.asarray(action)==LATE,KEEP,action),duration,strength)


def targets(item):
    labeled=annotate(item)
    y=np.full(len(item['events']),KEEP,dtype=np.int64)
    y[(labeled['y']==0)&labeled['mask']]=REMOVE
    for pitch in np.unique(item['events'][:,2]):
        indices=np.flatnonzero(item['events'][:,2]==pitch)
        if not np.all(item['eligible'][indices]&labeled['mask'][indices]&(labeled['y'][indices]==1)):
            continue
        action=np.full(len(y),KEEP,dtype=np.int64)
        action[indices]=EARLY
        new,kept=transform(item['events'],action,item['duration'])
        if np.array_equal(new,item['events']):
            continue
        clock=timing(item,new,kept)
        gain=clock['before_onset_release_error_sum'][0]-clock['after_onset_release_error_sum'][0]
        if gain>.018*len(indices) and clock['passes'] and protected(item,new)[0]:
            y[indices]=EARLY
    return {**labeled,'action_y':y}
