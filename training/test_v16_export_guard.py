"""A portable V16 export cannot use a failed, retuned, or replaced first pass."""

import json

import bass_temporal_v16_evidence as evidence
import pytest
from prepare_robust_training_stems import digest
from train_bass_temporal_v16 import FRESH_SEEDS


@pytest.mark.parametrize('change', [None, 'no_gain', 'lost_note', 'threshold', 'checkpoint', 'retuning',
                                   'incomplete', 'replacement', 'source_group', 'piano_result'])
def test_portable_export_requires_the_single_positive_preserved_first_pass(tmp_path, monkeypatch, change):
    run, fresh = tmp_path / 'run', tmp_path / 'fresh'
    run.mkdir()
    fresh.mkdir()
    for name in ('batch-selection.json', 'consumed-regression.json', 'piano-regression.json'):
        (run / name).write_text('{}')
    winner = {'checkpoint_sha256': 'frozen', 'threshold': .2, 'guardian_threshold': .3}
    ids = ['bass-temporal-v16-fresh-v1-' + str(seed) for seed in FRESH_SEEDS]
    plan = {'seeds': list(FRESH_SEEDS), 'positive_first_pass_gain_required': True,
            'checkpoint_sha256': 'frozen', 'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'consumed_regression_sha256': digest(run / 'consumed-regression.json'),
            'piano_regression_sha256': digest(run / 'piano-regression.json'), 'code_sha256': {}}
    manifest = {'now_consumed_regression': True, 'no_fitting_selection_user_audio_or_scores': True,
                'items': [{'id': name, 'source_group': 'original-bass-seed-' + str(seed)}
                          for name, seed in zip(ids, FRESH_SEEDS, strict=True)]}
    result = {'passes': True, 'false_notes_removed': 2, 'first_pass_complete': True,
              'now_consumed_regression': True, 'no_retuning': True, **winner,
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'per_recording': [{'id': name, 'passes': True} for name in ids]}
    if change == 'no_gain':
        result['false_notes_removed'] = 0
    elif change == 'lost_note':
        result['per_recording'][0]['passes'] = False
    elif change == 'threshold':
        result['threshold'] = .1
    elif change == 'checkpoint':
        result['checkpoint_sha256'] = 'runner-up'
    elif change == 'retuning':
        result['no_retuning'] = False
    elif change == 'incomplete':
        result['per_recording'].pop()
    elif change == 'replacement':
        manifest['items'][0]['id'] = 'different-fixture'
    elif change == 'source_group':
        manifest['items'][0]['source_group'] = 'previously-fitted-source'
    elif change == 'piano_result':
        plan['piano_regression_sha256'] = 'changed'
    (fresh / 'plan.json').write_text(json.dumps(plan))
    manifest['plan_sha256'] = digest(fresh / 'plan.json')
    (fresh / 'manifest.json').write_text(json.dumps(manifest))
    result['test_manifest_sha256'] = digest(fresh / 'manifest.json')
    (run / 'original-first-pass.json').write_text(json.dumps(result))
    monkeypatch.setattr(evidence, 'cached_items', lambda folder, manifest: manifest['items'])
    if change is None:
        assert len(evidence.fresh_items(run, fresh, winner)) == 32
    else:
        with pytest.raises(ValueError, match='first-pass'):
            evidence.fresh_items(run, fresh, winner)
