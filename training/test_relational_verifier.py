"""The experiment must compare with V4, rather than claiming its earlier gains."""

from types import SimpleNamespace

import numpy as np
import train_relational_verifier as relational


def test_preparation_applies_existing_v4_before_learning_new_relationships(monkeypatch):
    source = {'events': np.array([[0., 1., 60., 80.], [1., 2., 64., 80.], [2., 3., 67., 80.]]),
              'x': np.zeros((3, 52), dtype=np.float32), 'p': np.array([.2, .2, .8]),
              'keep': np.array([True, False, True]), 'shared': np.ones(3, dtype=bool), 'seconds': 5.}
    monkeypatch.setattr(relational, 'prepare_items', lambda *args: [source])
    verifier = SimpleNamespace(residual_model=SimpleNamespace(
        threshold=.0375, probability=lambda _: np.array([.01, .9, .01])))
    result = relational.prepare(None, [], verifier)[0]
    np.testing.assert_array_equal(result['keep'], [False, False, True])
    assert result['x'].shape == (3, 72)


def test_no_gain_is_claimed_for_notes_v4_already_removed():
    item = {'id': 'baseline', 'corpus': 'controlled', 'seconds': 2.,
            'events': np.array([[0., 1., 60., 80.], [1., 2., 75., 80.]]),
            'reference': np.array([[0., 1., 60.]]),
            'keep': np.array([True, False]), 'shared': np.ones(2, dtype=bool),
            'p': np.array([.4, .4])}
    result = relational.score([item], [np.array([.9, .001])], .01)
    assert result['passes'] and result['false_notes_removed'] == 0
    assert result['per_recording'][0]['deployed_v4']['false_positives'] == 0
    assert 'deployed_v3' not in result['per_recording'][0]


def test_uncertain_edge_notes_are_protected_when_context_is_missing(monkeypatch):
    source = {'id': 'crop-edge', 'corpus': 'controlled', 'seconds': 2.,
              'events': np.array([[.25, 1.5, 60., 80.]]),
              'reference': np.array([[.25, 1.5, 60.]]),
              'x': np.zeros((1, 52), dtype=np.float32), 'p': np.array([.2]),
              'keep': np.array([True]), 'shared': np.array([True])}
    monkeypatch.setattr(relational, 'prepare_items', lambda *args: [source])
    verifier = SimpleNamespace(residual_model=SimpleNamespace(
        threshold=.0375, probability=lambda _: np.array([.9])))
    item = relational.prepare(None, [], verifier)[0]
    assert not item['shared'].any()
    result = relational.score([item], [np.array([0.])], .025)
    assert result['passes']
    assert result['per_recording'][0]['candidate']['true_positives'] == 1
