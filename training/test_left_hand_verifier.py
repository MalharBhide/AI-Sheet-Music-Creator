"""Protect the actual V4 baseline, treble events, and valid sustained notes."""

from types import SimpleNamespace

import numpy as np
import train_left_hand_verifier as left


def test_low_register_training_starts_from_v4_and_protects_treble(monkeypatch):
    source = {'events': np.array([[0., 1., 48., 80.], [1., 2., 50., 80.], [2., 3., 60., 80.]]),
              'x': np.zeros((3, 52), dtype=np.float32), 'p': np.array([.2, .2, .2]),
              'keep': np.array([True, True, True]), 'shared': np.ones(3, dtype=bool)}
    monkeypatch.setattr(left, 'prepare_items', lambda *args: [source])
    verifier = SimpleNamespace(residual_model=SimpleNamespace(
        threshold=.0375, probability=lambda _: np.array([.01, .9, .9])))
    result = left.prepare(None, [], verifier)[0]
    np.testing.assert_array_equal(result['keep'], [False, True, True])
    np.testing.assert_array_equal(result['shared'], [True, True, False])
    np.testing.assert_array_equal(left.residual_keep(result, np.zeros(3), .01), [False, False, True])


def test_gate_catches_lost_hold_even_when_its_attack_still_has_a_match():
    item = {'id': 'bass-hold', 'corpus': 'controlled', 'seconds': 4.,
            'reference': np.array([[0., 3., 48.]]),
            'events': np.array([[0., 3., 48., 80.], [0., .2, 48., 80.], [1., 2., 53., 80.]]),
            'keep': np.ones(3, dtype=bool), 'shared': np.ones(3, dtype=bool), 'p': np.full(3, .2)}
    result = left.score([item], [np.array([0., 1., 0.])], .01)
    assert result['false_notes_removed'] == 2
    assert result['per_recording'][0]['matched_references_preserved']
    assert not result['per_recording'][0]['held_references_preserved']
    assert not result['passes']
    safe = left.score([item], [np.array([1., 0., 0.])], .01)
    assert safe['passes'] and safe['false_notes_removed'] == 2
