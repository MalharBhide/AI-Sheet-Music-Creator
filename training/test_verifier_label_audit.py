import numpy as np
import pytest
from audit_verifier_labels import audit


def item(piece, matched):
    return {'id': f'vienna-{piece}_p01', 'corpus': 'vienna-piano',
            'y': np.array([1] * matched + [0] * (100 - matched)),
            'events': np.zeros((100, 4)), 'reference': np.zeros((100, 3))}


def test_good_average_cannot_hide_a_misaligned_composition():
    with pytest.raises(ValueError, match='clock failure'):
        audit([item('Mozart', 100)] * 10 + [item('Chopin', 0)], [])


def test_both_training_and_validation_clocks_are_checked():
    with pytest.raises(ValueError, match='clock failure'):
        audit([item('Chopin', 90)], [item('Chopin', 0)])
    assert len(audit([item('Chopin', 90)], [item('Chopin', 80)])) == 2
