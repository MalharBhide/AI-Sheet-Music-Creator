"""Learn rearticulation without erasing genuine repeats, rests or uncertain timing."""

import numpy as np
from repeat_boundary_supervision import boundaries, labels


def item(events):
    return {'events': np.asarray(events, dtype=float), 'keep': np.ones(len(events), dtype=bool),
            'shared': np.ones(len(events), dtype=bool)}


def test_contiguous_fragments_of_a_labeled_hold_are_negative_attacks():
    data = item([[3., 4., 48., 80.], [4., 5., 48., 70.], [5., 6., 48., 90.]])
    pairs = boundaries(data)
    np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])
    y, mask = labels(np.array([[3., 6., 48.]]), data['events'], pairs)
    assert y.tolist() == [0., 0.] and mask.tolist() == [True, True]


def test_real_repeats_are_positive_even_when_other_reference_holds_overlap_them():
    data = item([[3., 4., 48., 80.], [4., 5., 48., 90.]])
    y, mask = labels(np.array([[3., 6., 48.], [4., 5., 48.]]), data['events'], boundaries(data))
    assert y.tolist() == [1.] and mask.tolist() == [True]


def test_gap_overlap_unshared_and_short_pairs_do_not_enter_supervision():
    for previous_end, current_start in [(3.5, 4.), (4.1, 4.), (3.1, 3.1)]:
        data = item([[3., previous_end, 48., 80.], [current_start, 5., 48., 80.]])
        assert boundaries(data).shape == (0, 2)
    data = item([[3., 4., 48., 80.], [4., 5., 48., 80.]])
    data['shared'][0] = False
    assert len(boundaries(data)) == 0


def test_nearby_annotation_real_rest_and_extended_release_stay_ambiguous():
    data = item([[3., 4., 48., 80.], [4., 5., 48., 80.]])
    for reference in (np.array([[3., 4., 48.], [4.1, 5., 48.]]),
                      np.array([[3., 3.5, 48.], [4.5, 5., 48.]]),
                      np.array([[3., 4.5, 48.]])):
        _, mask = labels(reference, data['events'], boundaries(data))
        assert not mask.any()


def test_removed_intermediate_attacks_do_not_supply_boundary_features():
    data = item([[3., 4., 48., 80.], [4., 4.5, 48., 80.], [4.5, 5., 48., 80.]])
    data['keep'][1] = False
    assert len(boundaries(data)) == 0
