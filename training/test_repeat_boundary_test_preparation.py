import pytest
from prepare_repeat_boundary_test import select


def test_boundary_stress_sources_use_fixed_test_performer_only():
    manifest = {'tracks': {'train': [{'id': 'must-not-read'}], 'validation': [], 'test': []}}
    for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS'):
        for role in ('comp', 'solo'):
            for index in range(3):
                manifest['tracks']['test'].append({'id': f'05_{genre}{index}-{role}',
                                                   'corpus': 'guitarset', 'player': '05'})
    selected = select(manifest)
    assert len(selected) == 10
    assert all('1-' in item['id'] and item['group'] == 'test' and item['player'] == '05' for item in selected)
    manifest['tracks']['test'][1]['player'] = '04'
    with pytest.raises(ValueError, match='reserved test performer'):
        select(manifest)
