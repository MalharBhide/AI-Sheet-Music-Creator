"""A safe but zero-gain stricter test still cannot bypass the frozen gate."""

import json

import pytest
from release_early_consensus_v22 import export


@pytest.mark.parametrize('bad_configuration',['zero_gain','changed_strength'])
def test_first_pass_failure_never_writes_portable_weights(tmp_path,monkeypatch,bad_configuration):
    import release_early_consensus_v22 as module

    run,fresh,output=tmp_path/'run',tmp_path/'fresh',tmp_path/'must-not-exist'
    run.mkdir()
    fresh.mkdir()
    winner={'checkpoint_sha256':'checkpoint','timing_threshold':.8,'remove_threshold':1.01}
    plan={'fresh_seeds':list(range(1,33)),'baseline_hashes':{}}
    (run/'batch-selection.json').write_text('{}')
    prepared={'seeds':list(range(1,33)),'baseline_hashes':{},
              'batch_selection_sha256':module.digest(run/'batch-selection.json'),'checkpoint_sha256':'checkpoint','consumed_regression_sha256':'regression',
              'no_fitting_selection_user_audio_or_scores':True,'code_sha256':{'training/fresh_early_consensus_v22.py':module.digest(module.Path(__file__).with_name('fresh_early_consensus_v22.py'))}}
    (fresh/'plan.json').write_text(json.dumps(prepared))
    manifest={'plan_sha256':module.digest(fresh/'plan.json'),'items':[{'id':'fixture-'+str(seed),'source_group':'original-bass-seed-'+str(seed),'group':'test'} for seed in range(1,33)],
              'now_consumed_regression':True,'no_fitting_selection_user_audio_or_scores':True}
    (fresh/'manifest.json').write_text(json.dumps(manifest))
    rows=[{'passes':True}]*32
    raw={'passes':True,'per_recording':rows,'onset_error_reduction_seconds':.02,
         'timing_threshold':.8,'remove_threshold':1.01,'strength':1.}
    margin={**raw,'timing_threshold':.9,'strength':.5,'onset_error_reduction_seconds':.01}
    if bad_configuration=='zero_gain':
        margin['onset_error_reduction_seconds']=0.
    else:
        margin['strength']=1.
    report={'passes':True,'first_pass_complete':True,'no_retuning':True,'now_consumed_regression':True,
            'recordings':32,'checkpoint_sha256':'checkpoint','batch_selection_sha256':module.digest(run/'batch-selection.json'),
            'test_manifest_sha256':module.digest(fresh/'manifest.json'),'raw':raw,'margin':margin}
    (run/'original-first-pass.json').write_text(json.dumps(report))
    monkeypatch.setattr(module,'frozen',lambda _: (plan,winner,None,None))
    monkeypatch.setattr(module,'require_regression',lambda *_:'regression')
    with pytest.raises(ValueError,match='Changed first-pass decision'):
        export(run,fresh,output)
    assert not output.exists()
