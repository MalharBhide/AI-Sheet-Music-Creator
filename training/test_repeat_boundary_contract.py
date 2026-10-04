"""Verify the website's label-free boundary contract against frozen training."""

import numpy as np
from app.services import repeat_boundary as production
from repeat_boundary_features import features, merge
from repeat_boundary_supervision import boundaries
from test_repeat_boundaries import fixture


def test_runtime_contract_matches_frozen_pairing_features_and_chain_merging():
    item = fixture()
    pairs = boundaries(item)
    np.testing.assert_array_equal(production.boundaries(item), pairs)
    for acoustic, names in ((False, production.RELATION_NAMES), (True, production.ACOUSTIC_NAMES)):
        expected = features(item, pairs, acoustic=acoustic)
        assert expected.shape[1] == len(names)
        np.testing.assert_array_equal(production.features(item, pairs, acoustic=acoustic), expected)
    for p, g in [(np.zeros(2), np.zeros(2)), (np.ones(2), np.zeros(2)),
                 (np.array([.05, 0.]), np.array([0., .025]))]:
        expected, merged = merge(item, pairs, p, g, .05, .025)
        actual, plan = production.merge(item, pairs, p, g, .05, .025)
        np.testing.assert_array_equal(actual, expected)
        assert plan == merged
