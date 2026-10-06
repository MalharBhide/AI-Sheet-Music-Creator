"""Exclude regression audio and invalid source partitions before observing waveforms."""

import pytest
from prepare_release_local_bass import prepare
from release_local_bass_data import load


def test_fitting_loader_refuses_test_partition_before_reading_files(tmp_path):
    with pytest.raises(ValueError,match='cannot read regression'):
        load(tmp_path,('test',))


@pytest.mark.parametrize('damage',['count','group','duplicate','overlap'])
def test_preparation_rejects_bad_split_before_audio_or_output(tmp_path,monkeypatch,damage):
    import prepare_release_local_bass as module
    items=[{'id':str(n),'source_group':str(n),'group':'train' if n<573 else 'validation'} for n in range(733)]
    if damage=='count':
        items.pop()
    elif damage=='group':
        items[0]['group']='test'
    elif damage=='duplicate':
        items[-1]['id']=items[0]['id']
    else:
        items[-1]['source_group']=items[0]['source_group']
    monkeypatch.setattr(module,'load',lambda _:items)
    monkeypatch.setattr(module.sf,'read',lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('Invalid partition reached audio')))
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError,match='Invalid release-local fitting partition'):
        prepare(tmp_path/'parent',output)
    assert not output.exists()
