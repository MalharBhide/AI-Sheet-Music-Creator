"""A positive validation candidate cannot bypass the consumed regression gate."""

import json

import pytest
from evaluate_v16_piano_regression import old_regression
from prepare_robust_training_stems import digest


@pytest.mark.parametrize('change', [None, 'recording', 'threshold', 'checkpoint', 'count', 'no_gain'])
def test_reserved_piano_requires_the_same_positive_preserved_frozen_candidate(tmp_path, change):
    (tmp_path / 'batch-selection.json').write_text('{"frozen": true}')
    winner = {'checkpoint_sha256': 'frozen-weights', 'threshold': .2, 'guardian_threshold': .3}
    result = {'passes': True, 'false_notes_removed': 18, 'per_recording': [{'passes': True} for _ in range(252)],
              'checkpoint_sha256': winner['checkpoint_sha256'], 'batch_selection_sha256': digest(tmp_path / 'batch-selection.json'),
              'threshold': .2, 'guardian_threshold': .3}
    if change == 'recording':
        result['per_recording'][0]['passes'] = False
    elif change == 'threshold':
        result['threshold'] = .1
    elif change == 'checkpoint':
        result['checkpoint_sha256'] = 'alternate-model'
    elif change == 'count':
        result['per_recording'].pop()
    elif change == 'no_gain':
        result['false_notes_removed'] = 0
    path = tmp_path / 'consumed-regression.json'
    path.write_text(json.dumps(result))
    if change is None:
        assert old_regression(tmp_path, winner) == digest(path)
    else:
        with pytest.raises(ValueError, match='frozen consumed regression'):
            old_regression(tmp_path, winner)
