"""Clarify phantom repeats using labeled holds whose attack is already matched.

No annotation is changed. Missing initial attacks and nearby genuine repeats stay
ambiguous, so an isolated late detection is never taught to be a false note.
"""

import numpy as np
from cache_note_verifier import targets

VERSION = 'covered-hold-repeats-v2'


def clarify(reference, events):
    import librosa
    import mir_eval

    reference, events = np.asarray(reference), np.asarray(events)
    y, mask = targets(reference, events)
    if not len(reference) or not len(events):
        return y, mask, np.zeros(len(events), dtype=bool)
    matches = mir_eval.transcription.match_notes(reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]), onset_tolerance=.05, offset_ratio=None)
    represented = reference[[r for r, _ in matches]]
    detections = events[[e for _, e in matches]]
    clarified = np.zeros(len(events), dtype=bool)
    for index in np.flatnonzero((y == 0) & ~mask):
        start, end, pitch, _ = events[index]
        same = np.abs(reference[:, 2] - pitch) <= .5
        # Keep every ambiguous detection within the original .15s attack
        # tolerance, including an actual repeat near a previous held note.
        if np.any(same & (np.abs(reference[:, 0] - start) <= .15)):
            continue
        covered = (np.abs(detections[:, 2] - pitch) <= .5) & (detections[:, 0] <= start)
        covered &= detections[:, 1] >= end
        held = represented[covered & (np.abs(represented[:, 2] - pitch) <= .5)
                           & (represented[:, 0] + .15 < start)
                           & (start < represented[:, 1] - .05)
                           & (end <= represented[:, 1] + .05)]
        if len(held):
            clarified[index] = True
    return y, mask | clarified, clarified
