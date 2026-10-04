import numpy as np
from current_accompaniment_baseline import advance
from test_left_consensus_v8 import fixture


def test_new_training_baseline_includes_v8_without_restoring_notes_or_changing_events():
    item = fixture()
    item['keep'][0] = False
    item['p'][:] = .9
    original = item['events'].copy()
    advance(item, np.zeros(3), np.array([0., .05, 0.]))
    assert item['keep'].tolist() == [False, True, False]
    np.testing.assert_array_equal(item['baseline_keep'], item['keep'])
    np.testing.assert_array_equal(item['events'], original)
    assert not item['mask'][0] and not item['mask'][2]
