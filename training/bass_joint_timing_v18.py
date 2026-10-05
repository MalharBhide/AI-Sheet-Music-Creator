"""Align learned timing labels with attack, hold, and repeated-spacing gates."""

import numpy as np
from bass_joint_timing import EARLY, KEEP, LATE, transform
from bass_joint_timing import targets as previous_targets
from score_bass_joint_timing import evaluate, timing

VERSION = 'bass-v16-joint-attack-correction-v2'


def margin(threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1.01:
        raise ValueError('Invalid action confidence threshold')
    # Halve the remaining probability margin, not an absolute increment which
    # disables all corrections at the conservative .99 confidence threshold.
    return threshold+(1.-threshold)*.5 if threshold <= 1. else 1.01


def targets(item):
    result = previous_targets(item)
    y = result['action_y'].copy()
    for index in np.flatnonzero((y==EARLY)|(y==LATE)):
        action = np.full(len(y),KEEP,np.int64)
        action[index] = y[index]
        new,indices = transform(item['events'],action,item['duration'])
        if not timing(item,new,indices)['passes']:
            y[index] = KEEP
    return {**result,'action_y':y}


def choose(items, probabilities, guardians, thresholds, memo):
    eligible,search = [],[]
    for timing_threshold in thresholds:
        for remove_threshold in thresholds:
            raw = evaluate(items,probabilities,guardians,timing_threshold,remove_threshold,memo=memo)
            guarded = evaluate(items,probabilities,guardians,margin(timing_threshold),margin(remove_threshold),.5,memo)
            passes = (raw['passes'] and guarded['passes']
                      and raw['onset_error_reduction_seconds']>1e-6
                      and guarded['onset_error_reduction_seconds']>1e-6)
            summary = {k:raw[k] for k in ('timing_threshold','remove_threshold','passes','failed_recordings',
                       'retimed_notes','false_notes_removed','onset_error_reduction_seconds')}
            summary.update(eligible=bool(passes),margin_passes=guarded['passes'],
                           margin_failed_recordings=guarded['failed_recordings'],
                           margin_onset_error_reduction_seconds=guarded['onset_error_reduction_seconds'])
            search.append(summary)
            if passes:
                key = (raw['onset_error_reduction_seconds'],raw['false_notes_removed'],timing_threshold,remove_threshold)
                eligible.append((key,{'raw':raw,'margin':guarded}))
    return max(eligible,key=lambda pair:pair[0])[1] if eligible else None,search
