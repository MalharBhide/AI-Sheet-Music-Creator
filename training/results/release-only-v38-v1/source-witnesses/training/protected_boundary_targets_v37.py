"""Training-only abstention labels; never expose reference support to inference."""

import numpy as np
from chord_boundary_v36 import SHIFTS, STAY, labels, transform
from pitch_interval_coverage import reference_coverage
from train_melody_timing import matches


def protected_labels(item):
    result = labels(item)
    events, reference, support = [item[k] for k in ("events", "reference", "pitch_reference")]
    attack = {i: r for r, i in matches(reference, events)}
    hold = {i: r for r, i in matches(reference, events, offsets=True)}
    loose = {i: r for r, i in matches(reference, events, .2)}
    zeros = np.zeros(len(events))
    remove = np.zeros(len(events), bool)
    for index in np.flatnonzero(item["eligible"]):
        same = events[:, 2] == events[index, 2]
        pitch_support = support[np.abs(support[:, 2]-events[index, 2]) <= .5]
        before_coverage = reference_coverage(pitch_support, events[same])
        for side in (0, 1):
            target = result[index, side+1]
            if target in (-100, STAY):
                continue
            proposal = zeros.copy()
            proposal[index] = SHIFTS[target]
            after, _ = transform(events, proposal if side == 0 else zeros,
                                 proposal if side == 1 else zeros, remove, item["duration"])
            safe = after[index, side] != events[index, side]
            safe &= np.all(reference_coverage(pitch_support, after[same]) >= before_coverage-1e-6)
            if index in attack:
                safe &= abs(after[index, 0]-reference[attack[index], 0]) <= .050000001
            if index in hold:
                r = reference[hold[index]]
                safe &= abs(after[index, 1]-r[1]) <= max(.05, .2*(r[1]-r[0]))+1e-9
            if index in loose:
                r = loose[index]
                safe &= abs(after[index, side]-reference[r, side]) <= abs(events[index, side]-reference[r, side])+1e-9
                if side == 0:
                    # Protect each observed repeat/chord relation, independently
                    # of whether other notes will also receive a correction.
                    for other, ref in loose.items():
                        if other == index:
                            continue
                        is_repeat = reference[r, 2] == reference[ref, 2]
                        is_chord = abs(reference[r, 0]-reference[ref, 0]) <= .025
                        if is_repeat or is_chord:
                            gap = reference[r, 0]-reference[ref, 0]
                            safe &= abs(after[index, 0]-events[other, 0]-gap) <= abs(events[index, 0]-events[other, 0]-gap)+1e-9
            # Extending a release must not manufacture unsupported sustain.
            annotated = np.column_stack((pitch_support, np.ones(len(pitch_support))))
            old_support = reference_coverage(events[index:index+1, :3], annotated)[0]
            new_support = reference_coverage(after[index:index+1, :3], annotated)[0]
            old_excess = events[index, 1]-events[index, 0]-old_support
            new_excess = after[index, 1]-after[index, 0]-new_support
            safe &= new_excess <= old_excess+1e-6
            if not safe:
                result[index, side+1] = STAY
    return result
