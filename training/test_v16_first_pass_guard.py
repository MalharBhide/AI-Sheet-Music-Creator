"""Fresh fixtures require preserved, complete evidence for the frozen winner."""

import json
from pathlib import Path

import pytest
from fresh_bass_temporal_v16 import require_regression
from prepare_robust_training_stems import digest


def evidence(tmp_path):
    run = tmp_path / 'candidate'
    run.mkdir()
    (run / 'batch-selection.json').write_text('{"frozen": true}')
    winner = {'checkpoint_sha256': 'one-frozen-winner', 'threshold': .2, 'guardian_threshold': .3}
    shared = {'passes': True, 'false_notes_removed': 18, **winner,
              'batch_selection_sha256': digest(run / 'batch-selection.json')}
    old = {**shared, 'per_recording': [{'passes': True} for _ in range(252)]}
    (run / 'consumed-regression.json').write_text(json.dumps(old))
    folder = tmp_path / 'v16-piano-regression-v1'
    folder.mkdir()
    plan = {'sources': [{'id': 'piano-performance'}], 'variants': ['piano-mp3', 'demucs-bass'],
            'checkpoint_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': shared['batch_selection_sha256'],
            'prior_regression_sha256': digest(run / 'consumed-regression.json'),
            'producer_sha256': digest(Path(__file__).with_name('evaluate_v16_piano_regression.py')),
            'no_fitting_selection_or_user_audio': True}
    (folder / 'plan.json').write_text(json.dumps(plan))
    item_id = 'v16-piano-regression-piano-performance-piano-mp3'
    manifest = {'plan_sha256': digest(folder / 'plan.json'), 'now_consumed_regression': True,
                'items': [{'id': item_id}], 'excluded_no_bass': [
                    {'id': 'v16-piano-regression-piano-performance-demucs-bass',
                     'excluded_no_bass': True, 'plan_sha256': digest(folder / 'plan.json')}]}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    piano = {**shared, 'false_notes_removed': 0, 'per_recording': [{'id': item_id, 'passes': True}],
             'all_declared_sources_checked': True, 'no_retuning_or_replacement': True,
             'now_consumed_regression': True, 'manifest_sha256': digest(folder / 'manifest.json')}
    (run / 'piano-regression.json').write_text(json.dumps(piano))
    return run, winner, old, folder, plan, manifest, piano


@pytest.mark.parametrize('change', [None, 'old-count', 'old-no-gain', 'old-recording', 'old-threshold',
                                   'piano-recording', 'piano-checkpoint', 'piano-threshold', 'piano-sources',
                                   'piano-retuned', 'piano-unconsumed', 'manifest-sha', 'missing-source',
                                   'duplicate-source', 'wrong-result-id', 'producer', 'prior-evidence'])
def test_first_pass_cannot_bypass_or_replace_reserved_regression(tmp_path, change):
    run, winner, old, folder, plan, manifest, piano = evidence(tmp_path)
    if change == 'old-count':
        old['per_recording'].pop()
    elif change == 'old-no-gain':
        old['false_notes_removed'] = 0
    elif change == 'old-recording':
        old['per_recording'][0]['passes'] = False
    elif change == 'old-threshold':
        old['threshold'] = .1
    elif change == 'piano-recording':
        piano['per_recording'][0]['passes'] = False
    elif change == 'piano-checkpoint':
        piano['checkpoint_sha256'] = 'runner-up'
    elif change == 'piano-threshold':
        piano['threshold'] = .1
    elif change == 'piano-sources':
        piano['all_declared_sources_checked'] = False
    elif change == 'piano-retuned':
        piano['no_retuning_or_replacement'] = False
    elif change == 'piano-unconsumed':
        piano['now_consumed_regression'] = False
    elif change == 'manifest-sha':
        piano['manifest_sha256'] = 'changed'
    elif change == 'missing-source':
        manifest['excluded_no_bass'] = []
    elif change == 'duplicate-source':
        manifest['excluded_no_bass'].append(manifest['excluded_no_bass'][0].copy())
    elif change == 'wrong-result-id':
        piano['per_recording'][0]['id'] = 'replacement-recording'
    elif change == 'producer':
        plan['producer_sha256'] = 'changed-producer'
    elif change == 'prior-evidence':
        plan['prior_regression_sha256'] = 'alternate-result'
    (run / 'consumed-regression.json').write_text(json.dumps(old))
    (folder / 'plan.json').write_text(json.dumps(plan))
    # Rebind envelope hashes so structural checks, not stale hashes alone,
    # reject missing/duplicate source coverage and changed frozen provenance.
    manifest['plan_sha256'] = digest(folder / 'plan.json')
    for item in manifest['excluded_no_bass']:
        item['plan_sha256'] = manifest['plan_sha256']
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    if change != 'manifest-sha':
        piano['manifest_sha256'] = digest(folder / 'manifest.json')
    (run / 'piano-regression.json').write_text(json.dumps(piano))
    if change is None:
        assert require_regression(run, winner) == digest(run / 'consumed-regression.json')
    else:
        with pytest.raises(ValueError, match='regression|sources'):
            require_regression(run, winner)
