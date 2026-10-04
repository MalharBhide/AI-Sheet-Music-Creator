"""Protect crop-context parity and the actual deployed V6 baseline."""

import numpy as np
import pytest
from train_left_relations_v7 import (
    combine_scores,
    complete_evidence,
    enrich,
    predict,
    score,
    window_eligible,
)


def test_neighbors_outside_label_crop_are_present_in_target_features():
    events = np.array([[14., 18., 36., 90.], [15.5, 18., 48., 75.], [18., 20., 50., 70.]])
    x, p = np.zeros((3, 26)), np.array([.9, .2, .4])
    target = {'events': events[1:], 'x': np.zeros((2, 52)), 'p': p[1:]}
    cropped, eligible = enrich(target, events, x, p, 30.)
    full, _ = enrich({'events': events, 'x': np.zeros((3, 52)), 'p': p}, events, x, p, 30.)
    np.testing.assert_array_equal(cropped, full[1:])
    assert eligible.all()
    incomplete, _ = enrich(target, events[1:], x[1:], p[1:], 30.)
    assert not np.array_equal(cropped[0], incomplete[0])


def test_small_cache_edge_crops_do_not_change_eligible_features():
    events = np.array([[.1, 8., 36., 90.], [2.6, 4., 48., 75.],
                       [4., 5., 50., 70.], [7.4, 8., 55., 80.], [9.9, 10., 48., 70.]])
    x, p = np.zeros((5, 26)), np.array([.9, .2, .4, .2, .6])
    target = {'events': events[1:-1], 'x': np.zeros((3, 52)), 'p': p[1:-1]}
    full, valid = enrich(target, events, x, p, 10.)
    cropped, _ = enrich(target, events[1:-1], x[1:-1], p[1:-1], 10.)
    assert valid.all()
    np.testing.assert_array_equal(full, cropped)
    edge_events = np.array([[t, t + .1, 48., 80.] for t in (0., 2.5, 2.51, 7.49, 7.5, 9.9)])
    np.testing.assert_array_equal(window_eligible(edge_events, 10.), [False, False, True, True, False, False])


def test_large_cropped_cache_cannot_silently_lose_neighbors(tmp_path):
    with pytest.raises(ValueError, match='full neighboring context'):
        complete_evidence(tmp_path, {'legacy_x': np.zeros((1, 26)), 'evaluation_window': [15.25, 29.75]})


def test_large_crop_uses_original_full_cache_instead_of_cropped_legacy(tmp_path):
    full = np.array([[1., 2., 48., 80.], [16., 17., 50., 80.]])
    original = tmp_path / 'note-verifier-features'
    original.mkdir()
    np.savez(original / 'piano.npz', events=full, x=np.zeros((2, 26)))
    item = {'id': 'piano', 'legacy_x': np.ones((1, 26)),
            'legacy_events': full[1:], 'evaluation_window': [15.25, 29.75]}
    events, x = complete_evidence(tmp_path / 'context', item)
    np.testing.assert_array_equal(events, full)
    assert x.shape == (2, 26)


def test_consensus_requires_both_acoustic_and_context_models_to_reject():
    class Head:
        def __init__(self, width, probabilities):
            self.width, self.probabilities = width, np.asarray(probabilities)

        def predict_proba(self, x):
            assert x.shape == (3, self.width)
            return np.column_stack((1 - self.probabilities, self.probabilities))

    models = [(72, Head(72, [0., .9, 0.])), (52, Head(52, [.8, 0., 0.]))]
    np.testing.assert_array_equal(predict(models, np.zeros((3, 72))), [.8, .9, 0.])


def test_separate_view_thresholds_keep_agreement_and_half_margin_semantics():
    context, acoustic = np.array([.01, .08, .01, .01]), np.array([.1, .01, .4, .2])
    combined = combine_scores(context, acoustic, .075, .4)
    np.testing.assert_array_equal(combined < .075, [True, False, False, True])
    np.testing.assert_array_equal(combined < .0375, [True, False, False, False])
    np.testing.assert_allclose(combined, combine_scores(context, acoustic, .0375, .2))


def test_memoized_validation_has_identical_metrics_for_repeated_masks():
    item = {'id': 'notes', 'corpus': 'controlled', 'seconds': 2.,
            'reference': np.array([[0., 1., 48.]]),
            'events': np.array([[0., 1., 48., 80.], [1., 2., 52., 80.]]),
            'keep': np.ones(2, dtype=bool), 'baseline_keep': np.ones(2, dtype=bool),
            'shared': np.ones(2, dtype=bool), 'p': np.full(2, .2)}
    memo = {}
    for threshold in (.01, .1, .2, .3, .01):
        probabilities = [np.array([.8, .15])]
        assert score([item], probabilities, threshold, memo) == score([item], probabilities, threshold)
    assert len(memo) == 2


def test_v6_prior_rejections_and_correct_holds_are_protected():
    item = {'id': 'bass', 'corpus': 'controlled', 'seconds': 5.,
            'reference': np.array([[0., 4., 48.]]),
            'events': np.array([[0., 4., 48., 80.], [1., 2., 50., 80.], [2., 3., 52., 80.]]),
            'keep': np.array([True, False, True]), 'baseline_keep': np.array([True, False, True]),
            'shared': np.ones(3, dtype=bool), 'p': np.full(3, .2)}
    unchanged = score([item], [np.ones(3)], .05)
    assert unchanged['passes'] and unchanged['false_notes_removed'] == 0
    safe = score([item], [np.array([1., 1., 0.])], .05)
    assert safe['passes'] and safe['false_notes_removed'] == 1
    assert safe['per_recording'][0]['deployed_v6']['false_positives'] == 1
    lost = score([item], [np.zeros(3)], .05)
    assert not lost['passes'] and not lost['per_recording'][0]['held_references_preserved']


def test_portable_relation_contract_is_explicit_and_matches_training(tmp_path):
    from app.services.note_context_model import ContextNoteModel
    from export_context_verifier import export
    from note_relations import ALL_NAMES, RELATION_VERSION
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 72)).astype(np.float32)
    model = HistGradientBoostingClassifier(max_iter=4, min_samples_leaf=5).fit(x, x[:, 65] > 0)
    path = tmp_path / 'relations.npz'
    export(model, .0375, path, [{'x': x}], feature_names=ALL_NAMES, feature_version=RELATION_VERSION)
    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match='Incompatible'):
            ContextNoteModel(saved)
        portable = ContextNoteModel(saved, feature_names=ALL_NAMES, feature_version=RELATION_VERSION)
    np.testing.assert_allclose(portable.probability(x), model.predict_proba(x)[:, 1], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='input'):
        portable.probability(x[:, :52])
