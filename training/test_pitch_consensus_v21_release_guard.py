"""A failed candidate must never reach first-pass audio inference."""

import json

import pytest
from fresh_pitch_consensus_v21 import verify
from train_pitch_consensus_v21 import FRESH_SEEDS


@pytest.mark.parametrize('failed_configuration',['raw','margin'])
def test_failed_regression_stops_before_generating_or_transcribing_audio(tmp_path, monkeypatch, failed_configuration):
    import fresh_pitch_consensus_v21 as module

    run=tmp_path/'run'
    run.mkdir()
    winner={'checkpoint_sha256':'checkpoint','timing_threshold':.9,'remove_threshold':.95}
    (run/'batch-selection.json').write_text('{}')
    configuration={'passes':True,'per_recording':[{'passes':True}]*328,
                   'onset_error_reduction_seconds':.1,'timing_threshold':.9,'remove_threshold':.95,'strength':1.}
    raw=dict(configuration)
    margin={**configuration,'timing_threshold':.95,'remove_threshold':.975,'strength':.5}
    report={'passes':True,'recordings':328,'consumed_regression':True,
            'checkpoint_sha256':'checkpoint','batch_selection_sha256':module.digest(run/'batch-selection.json'),
            'test_used_for_selection':False,'no_user_audio_or_scores':True,'raw':raw,'margin':margin}
    report[failed_configuration]['per_recording']=[{'passes':False}]+[{'passes':True}]*327
    (run/'consumed-regression.json').write_text(json.dumps(report))
    plan={'fresh_seeds':list(FRESH_SEEDS),'fresh_families':{'original':16,'texture':16}}
    monkeypatch.setattr(module,'frozen',lambda _: (plan,winner,None,None))

    def forbidden(*args,**kwargs):
        raise AssertionError('Rejected model reached audio generation or transcription')

    for name in ('_GeneralEngine','original','texture','encode'):
        monkeypatch.setattr(module,name,forbidden)
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError,match='Changed consumed timing regression'):
        verify(run,output)
    assert not output.exists() and not (run/'original-first-pass.json').exists()
