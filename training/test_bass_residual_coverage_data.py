"""Partial true-pitch support stays audible; labels never enter X."""

import numpy as np
from bass_residual_coverage_data import protect
from bass_residual_v11_data import annotate
from test_bass_v11_baseline import fixture


def test_partial_pitch_support_becomes_protected_fitting_supervision():
    item = fixture()
    item['pitch_reference'] = np.vstack((item['pitch_reference'], [4.9, 5., 44.]))
    previous = annotate(item)
    assert previous['y'][2] == 0
    actual = protect(previous)
    assert actual['y'].tolist() == [1., 1., 1.] and actual['mask'].all()
    assert actual['partial_pitch_support_promoted'] == 1
    np.testing.assert_array_equal(actual['x'], previous['x'])
    np.testing.assert_array_equal(actual['events'], previous['events'])
    assert previous['y'][2] == 0  # frozen historical labels were not mutated


def test_unsupported_wrong_pitch_remains_negative_and_feature_values_unchanged():
    previous = annotate(fixture())
    actual = protect(previous)
    assert actual['y'].tolist() == [1., 1., 0.]
    assert actual['partial_pitch_support_promoted'] == 0
    np.testing.assert_array_equal(actual['x'], previous['x'])


def test_zero_length_contacts_do_not_create_pitch_coverage():
    item = fixture()
    item['pitch_reference'] = np.vstack((item['pitch_reference'], [5., 6., 44.]))
    actual = protect(annotate(item))
    assert actual['y'][2] == 0
