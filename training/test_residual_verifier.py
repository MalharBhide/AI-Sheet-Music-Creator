"""Regression guards for training a filter on the deployed model's errors."""

from types import SimpleNamespace

import numpy as np
import torch
import train_residual_verifier as residual


def test_residual_filter_cannot_restore_rejected_or_remove_protected_notes():
    item = {'keep': np.array([True, True, True, False, True]),
            'shared': np.array([True, False, True, True, True]),
            'p': np.array([.2, .2, .50001, .02, .5])}
    keep = residual.residual_keep(item, np.array([.001, .001, .001, .9, .01]), .01)
    np.testing.assert_array_equal(keep, [False, True, True, False, True])
    np.testing.assert_array_equal(item['keep'], [True, True, True, False, True])


def test_gate_rejects_losing_a_correct_hold_even_when_false_notes_improve():
    item = {'id': 'held-melody', 'corpus': 'controlled', 'seconds': 5.,
            'reference': np.array([[0., 4., 60.], [4., 5., 62.]]),
            'events': np.array([[0., 4., 60., 80.], [4., 5., 62., 80.],
                                [1., 2., 75., 70.], [2., 3., 76., 70.]]),
            'keep': np.ones(4, dtype=bool), 'shared': np.ones(4, dtype=bool),
            'p': np.full(4, .2)}
    result = residual.evaluate([item], [np.array([.001, .9, .001, .001])], .01)
    assert result['false_notes_removed'] == 2
    assert result['failed_recordings'] == ['held-melody']
    assert not result['passes']
    clean = residual.evaluate([item], [np.array([.9, .9, .001, .001])], .01)
    assert clean['passes'] and clean['false_notes_removed'] == 2


def test_preparation_uses_deployed_events_and_preserves_unshared_decisions(monkeypatch):
    x = np.zeros((3, 52), dtype=np.float32)
    bounded = np.array([[0., 1., 60., 80.], [1., 2., 64., 70.], [2., 3., 67., 80.]])
    legacy = bounded.copy()
    legacy[1, 3] += 1  # Different velocity means different decoder candidate.
    item = {'id': 'event-alignment', 'x': x, 'events': bounded,
            'legacy_x': x[:, :26], 'legacy_events': legacy,
            'reference': np.array([[0., 1., 60.], [1., 2., 64.], [2., 3., 67.]])}
    model = SimpleNamespace(mean=np.zeros(26), scale=np.ones(26), threshold=.1,
        model=lambda _: torch.logit(torch.tensor([.15, .15, .8])),
        context_models=[SimpleNamespace(threshold=.01, probability=lambda x: np.zeros(len(x)))] * 2)
    prepared = residual.prepare_items(None, [item], model)[0]
    np.testing.assert_array_equal(prepared['events'], legacy)
    np.testing.assert_array_equal(prepared['shared'], [True, False, True])
    np.testing.assert_array_equal(prepared['keep'], [False, True, True])
    np.testing.assert_array_equal(prepared['y'], [1, 1, 1])


def test_preparation_rejects_feature_drift():
    import pytest

    item = {'x': np.ones((1, 52), dtype=np.float32), 'events': np.array([[0., 1., 60., 80.]]),
            'legacy_x': np.zeros((1, 26), dtype=np.float32),
            'legacy_events': np.array([[0., 1., 60., 80.]])}
    with pytest.raises(AssertionError):
        residual.prepare_items(None, [item], None)
