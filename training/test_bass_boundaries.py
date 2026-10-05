"""Bass repeat repair must not erase real rhythms or resurrect V10 rejections."""

from types import SimpleNamespace

import bass_boundary_evidence as evidence
import current_bass_baseline as baseline
import numpy as np
import pytest
import train_bass_boundaries as training


def fixture():
    events = np.array([[3., 4., 40., 80.], [4., 5., 40., 70.], [5., 6., 40., 60.]])
    return {'id': 'original-hold', 'corpus': 'original-fixture', 'group': 'validation',
            'events': events, 'keep': np.ones(3, dtype=bool), 'baseline_keep': np.ones(3, dtype=bool),
            'eligible': np.ones(3, dtype=bool), 'shared': np.ones(3, dtype=bool),
            'baseline_p': np.full(3, .9), 'baseline_g': np.full(3, .8), 'seconds': 10.,
            'reference': np.array([[3., 6., 40.]]), 'pitch_reference': np.array([[3., 6., 40.]]),
            'x': np.arange(3 * 52, dtype=np.float32).reshape(3, 52) / 1000}


def test_pair_features_are_label_free_and_guardian_matches_training_view():
    item = fixture()
    pairs = evidence.boundaries(item)
    assert pairs.tolist() == [[0, 1], [1, 2]]
    x = evidence.features(item, pairs)
    assert x.shape == (2, 112)
    np.testing.assert_array_equal(evidence.acoustic_view(x), evidence.features(item, pairs, acoustic=True))
    item['reference'][:] = 0
    item['pitch_reference'][:] = 0
    np.testing.assert_array_equal(evidence.features(item, pairs), x)
    assert evidence.features(item, np.empty((0, 2), int)).shape == (0, 112)


def test_true_repeated_attacks_override_pedal_support_and_are_protected_by_gate():
    item = fixture()
    item['reference'] = item['events'][:, :3].copy()
    evidence.annotate([item])
    assert item['boundary_y'].tolist() == [1., 1.]
    assert item['boundary_mask'].all()
    result = training.score([item], [(np.zeros(2), np.zeros(2))], .01, .1)
    assert not result['passes']
    assert not result['per_recording'][0]['matched_attacks_preserved']
    assert result['per_recording'][0]['entire_pitch_time_union_preserved']


def test_split_hold_merges_without_timing_velocity_or_coverage_loss():
    item = evidence.annotate([fixture()])[0]
    before = item['events'].copy()
    assert item['boundary_y'].tolist() == [0., 0.] and item['boundary_mask'].all()
    after, merged = evidence.merge(item, item['pairs'], np.zeros(2), np.zeros(2), .01, .1)
    assert len(merged) == 2
    np.testing.assert_array_equal(after, [[3., 6., 40., 80.]])
    np.testing.assert_array_equal(item['events'], before)
    result = training.score([item], [(np.zeros(2), np.zeros(2))], .01, .1)
    assert result['passes'] and result['false_notes_removed'] == 2


@pytest.mark.parametrize('change', ['real-rest', 'edge', 'rejected', 'long-overlap'])
def test_pair_builder_protects_real_rests_edges_and_previously_rejected_notes(change):
    item = fixture()
    if change == 'real-rest':
        item['events'][1, 0] += .001
    elif change == 'edge':
        item['shared'][1] = False
    elif change == 'rejected':
        item['keep'][1] = False
    else:
        item['events'][0, 1] += .04
    assert [0, 1] not in evidence.boundaries(item).tolist()


def test_uncertain_nearby_attack_and_uncovered_tail_do_not_become_negative_labels():
    item = fixture()
    item['reference'] = np.array([[4.1, 5., 40.]])
    y, mask = evidence.labels(item['reference'], item['pitch_reference'], item['events'], np.array([[0, 1]]))
    assert not mask[0] and y[0] == 0
    item['reference'] = np.array([[3., 4., 40.]])
    item['pitch_reference'] = item['reference'].copy()
    _, mask = evidence.labels(item['reference'], item['pitch_reference'], item['events'], np.array([[0, 1]]))
    assert not mask[0]


def test_shipped_consensus_exact_thresholds_and_empty_rows_match_runtime(monkeypatch):
    def model(values, count):
        return SimpleNamespace(threshold=.1, feature_count=count, probability=lambda x: np.asarray(values)[:len(x)])
    fake = SimpleNamespace(models=[model([.09, .1, .09], 52), model([.09, .09, .1], 26)])
    monkeypatch.setattr(baseline.service, 'BassVerifier', lambda: fake)
    item = fixture()
    actual = baseline.prepare([item])[0]
    assert actual['baseline_keep'].tolist() == [False, True, True]
    assert item['keep'].all()
    for name in ('events', 'x', 'eligible'):
        item[name] = item[name][:0]
    assert baseline.prepare([item])[0]['keep'].shape == (0,)


def test_invalid_cache_evidence_cannot_be_scored_as_shipped_baseline():
    item = fixture()
    item['x'][0, 0] = np.nan
    with pytest.raises(ValueError, match='baseline evidence'):
        baseline.prepare([item])


def test_matrix_uses_only_labeled_pairs_and_requires_both_classes():
    held = evidence.annotate([fixture()])[0]
    repeats = fixture()
    repeats['id'] = 'original-repeats'
    repeats['reference'] = repeats['events'][:, :3].copy()
    evidence.annotate([repeats])
    repeats['boundary_mask'][1] = False
    x, y, weight, counts = training.matrix([held, repeats], 30.)
    assert x.shape == (3, 112) and y.tolist() == [0., 0., 1.]
    assert weight[-1] > weight[0] and len(counts) == 2
    with pytest.raises(ValueError, match='real repeats and split holds'):
        training.matrix([held], 30.)


def test_boundary_checkpoints_accept_their_declared_350_stage_without_mutating_trained_models():
    models = [SimpleNamespace(_predictors=[None] * 500) for _ in range(2)]
    staged = training.at_stage(models, 350)
    assert [len(model._predictors) for model in staged] == [350, 350]
    assert [len(model._predictors) for model in models] == [500, 500]
    with pytest.raises(ValueError, match='frozen boundary stages'):
        training.at_stage(models, 300)
    with pytest.raises(ValueError, match='frozen boundary stages'):
        training.at_stage([SimpleNamespace(_predictors=[None] * 200)], 350)
