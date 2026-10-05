"""Protect correctly detected key releases as well as new repeated attacks.

Pedal pitch support is not evidence that an already correct key release can be
erased. Labels use fitting annotations only; inference features stay unchanged.
"""

import numpy as np
from bass_boundary_evidence import annotate as pitch_annotate
from evaluate_context_correction import matched_references
from train_left_hand_verifier import held_matches

VERSION = 'bass-key-articulation-labels-v2'


def annotate(items):
    pitch_annotate(items)
    for item in items:
        count = len(item['pairs'])
        item['protected_key_offsets'] = np.zeros(count, dtype=bool)
        item['protected_key_attacks'] = np.zeros(count, dtype=bool)
        for index, (before, after) in enumerate(item['pairs']):
            original = item['events'][[before, after]]
            reference = item['reference'][np.abs(item['reference'][:, 2] - original[0, 2]) <= .5]
            if not len(reference):
                continue
            merged = original[:1].copy()
            merged[0, 1] = max(original[:, 1])
            lost_offset = not held_matches(reference, original).issubset(held_matches(reference, merged))
            lost_attack = not matched_references(reference, original).issubset(matched_references(reference, merged))
            item['protected_key_offsets'][index] = lost_offset
            item['protected_key_attacks'][index] = lost_attack
            if lost_offset or lost_attack:
                item['boundary_y'][index], item['boundary_mask'][index] = 1., True
    return items
