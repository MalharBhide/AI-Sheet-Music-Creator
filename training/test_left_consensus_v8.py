"""V8 must measure improvement over V7 and preserve each surviving matched note."""

from types import SimpleNamespace

import numpy as np
import pytest
from prepare_relational_piano import prepare as prepare_piano
from train_left_consensus_v8 import (
    advance_baseline,
    probabilities,
    regression_items,
    score,
)


def fixture():
    return {'id': 'notes', 'corpus': 'controlled', 'seconds': 5.,
            'reference': np.array([[0., 4., 48.]]),
            'events': np.array([[0., 4., 48., 80.], [0., 4., 48., 70.], [1., 2., 52., 80.]]),
            'x': np.zeros((3, 72)), 'p': np.full(3, .2),
            'shared': np.ones(3, dtype=bool), 'keep': np.ones(3, dtype=bool)}


def test_baseline_relabels_surviving_duplicate_after_v7_filter():
    relation = SimpleNamespace(threshold=.0375, probability=lambda _: np.array([0., 1., 1.]))
    guardian = SimpleNamespace(threshold=.3, probability=lambda _: np.array([0., 1., 1.]))
    item = advance_baseline(fixture(), relation, guardian)
    np.testing.assert_array_equal(item['keep'], [False, True, True])
    np.testing.assert_array_equal(item['y'], [0., 1., 0.])
    np.testing.assert_array_equal(item['mask'], [False, True, True])
    result = score([item], [np.array([0., 1., 1.])], .1)
    assert result['passes'] and result['false_notes_removed'] == 0
    assert result['per_recording'][0]['deployed_v7']['true_positives'] == 1
    assert 'deployed_v6' not in result['per_recording'][0]
    improved = score([item], [np.array([1., 1., 0.])], .1)
    assert improved['passes'] and improved['false_notes_removed'] == 1
    lost = score([item], [np.zeros(3)], .1)
    assert not lost['passes'] and not lost['per_recording'][0]['held_references_preserved']


def test_baseline_cannot_restore_prior_rejections_or_remove_strong_and_unshared_notes():
    item = fixture()
    item['keep'][0], item['p'][1], item['shared'][2] = False, .9, False
    relation = SimpleNamespace(threshold=.0375, probability=lambda _: np.zeros(3))
    guardian = SimpleNamespace(threshold=.3, probability=lambda _: np.zeros(3))
    item = advance_baseline(item, relation, guardian)
    np.testing.assert_array_equal(item['keep'], [False, True, True])


def test_offline_consensus_matches_exact_production_threshold_ties():
    class Head:
        def __init__(self, width, values):
            self.width, self.values = width, np.asarray(values)

        def predict_proba(self, x):
            assert x.shape == (4, self.width)
            return np.column_stack((1 - self.values, self.values))

    models = [Head(72, [0., .0375, 0., 1.]), Head(52, [0., 0., .3, 0.])]
    np.testing.assert_array_equal(probabilities(models, np.zeros((4, 72)), .0375, .3), [0., 1., 1., 1.])


def test_future_regression_includes_consumed_v7_fresh_passages(monkeypatch, tmp_path):
    import train_left_consensus_v8 as training

    names = []
    monkeypatch.setattr(training, 'load_data', lambda *_: [])
    def external(directory, manifest, base):
        names.append(manifest.parent.name)
        return []
    monkeypatch.setattr(training, 'cached_external', external)
    assert regression_items(tmp_path) == []
    assert 'relational-piano-fresh-90' in names


def test_preparation_rejects_offsets_crossing_training_and_test_roles(tmp_path):
    with pytest.raises(ValueError, match='Training offsets'):
        prepare_piano(tmp_path, fresh=True, training_offset=60)
    with pytest.raises(ValueError, match='reserved test'):
        prepare_piano(tmp_path, fresh=False, test_offset=120)
    with pytest.raises(ValueError, match='Training offsets'):
        prepare_piano(tmp_path, training_offset=120)
    assert list(tmp_path.iterdir()) == []


def test_export_rejects_evaluation_from_changed_frozen_selection(tmp_path):
    import hashlib
    import json

    from export_left_consensus_v8 import release_reports

    selection = {'checkpoint_sha256': 'frozen-model', 'threshold': .025, 'guardian_threshold': .2}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection))
    report = {'passes': True, 'checkpoint_sha256': 'frozen-model', 'threshold': .025,
              'guardian_threshold': .2, 'selection_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'false_notes_removed': 1}
    for stage in ('regression', 'fresh'):
        (tmp_path / (stage + '.json')).write_text(json.dumps(report))
    release_reports(tmp_path, selection)
    path.write_text(json.dumps({**selection, 'threshold': .05}))
    with pytest.raises(ValueError, match='changed release'):
        release_reports(tmp_path, selection)
    path.write_text(json.dumps(selection))
    (tmp_path / 'regression.json').write_text(json.dumps({**report, 'false_notes_removed': 0}))
    with pytest.raises(ValueError, match='improve'):
        release_reports(tmp_path, selection)


def test_shared_evidence_keeps_acoustic_columns_and_confidences_in_both_views():
    from app.services.left_baseline_evidence import acoustic_evidence, append_evidence

    x = np.arange(3 * 72, dtype=np.float32).reshape(3, 72) / 100
    relation, guardian = np.array([.1, .5, .9]), np.array([.8, .2, .3])
    full = append_evidence(x, relation, guardian)
    np.testing.assert_array_equal(full[:, :72], x)
    np.testing.assert_allclose(full[:, -2:], np.column_stack((relation, guardian)))
    view = acoustic_evidence(full)
    assert view.shape == (3, 54)
    np.testing.assert_array_equal(view[:, :52], x[:, :52])
    np.testing.assert_array_equal(view[:, -2:], full[:, -2:])
    assert append_evidence(np.empty((0, 72)), np.empty(0), np.empty(0)).shape == (0, 74)
    with pytest.raises(ValueError, match='frozen'):
        append_evidence(x, np.array([np.nan, .1, .2]), guardian)
    with pytest.raises(ValueError, match='frozen'):
        append_evidence(x, relation, np.array([1.1, .1, .2]))
    with pytest.raises(ValueError, match='acoustic'):
        acoustic_evidence(x)


def test_evidence_portable_heads_match_training_and_require_explicit_contracts(tmp_path):
    from app.services.left_baseline_evidence import (
        ACOUSTIC_EVIDENCE_NAMES,
        EVIDENCE_VERSION,
        RELATION_EVIDENCE_NAMES,
        acoustic_evidence,
        append_evidence,
    )
    from app.services.note_context_model import ContextNoteModel
    from export_context_verifier import export
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(161)
    x = append_evidence(rng.normal(size=(100, 72)), rng.random(100), rng.random(100))
    for names, features in [(RELATION_EVIDENCE_NAMES, x), (ACOUSTIC_EVIDENCE_NAMES, acoustic_evidence(x))]:
        model = HistGradientBoostingClassifier(max_iter=4, min_samples_leaf=5).fit(features, x[:, -1] > .5)
        path = tmp_path / f'head-{features.shape[1]}.npz'
        export(model, .1, path, [{'x': features}], feature_names=names, feature_version=EVIDENCE_VERSION)
        with np.load(path, allow_pickle=False) as saved:
            with pytest.raises(ValueError, match='Incompatible'):
                ContextNoteModel(saved)
            portable = ContextNoteModel(saved, feature_names=names, feature_version=EVIDENCE_VERSION)
        np.testing.assert_allclose(portable.probability(features), model.predict_proba(features)[:, 1], rtol=1e-12, atol=1e-12)


def test_broad_fit_restores_only_shared_interior_treble_eligibility(monkeypatch, tmp_path):
    from contextlib import nullcontext

    import train_left_consensus_v8 as training

    events = np.array([[3., 5., 48., 80.], [3.1, 5., 72., 80.],
                       [3.2, 5., 74., 80.], [2.5, 5., 76., 80.]])
    item = {'events': events, 'reference': events[:, :3], 'audio': 'cached.wav',
            'x': np.zeros((4, 72)), 'p': np.full(4, .2), 'keep': np.ones(4, dtype=bool),
            'shared': np.array([True, False, False, False])}
    raw = {'events': events[[0, 1, 3]]}
    monkeypatch.setattr(training, 'hashes', dict)
    monkeypatch.setattr(training, 'prepare_v6', lambda *_: [item])
    monkeypatch.setattr(training.np, 'load', lambda *_args, **_kwargs: nullcontext({}))
    thresholds = iter([.0375, .3])
    monkeypatch.setattr(training, 'ContextNoteModel', lambda *_args, **_kwargs: SimpleNamespace(
        threshold=next(thresholds), probability=lambda x: np.ones(len(x))))
    monkeypatch.setattr(training.sf, 'info', lambda _: SimpleNamespace(duration=10.))
    result = training.prepare(tmp_path, [raw], baseline_evidence=True, full_register=True)[0]
    np.testing.assert_array_equal(result['shared'], [True, True, False, False])
    np.testing.assert_array_equal(result['keep'], np.ones(4, dtype=bool))
    assert result['x'].shape == (4, 74)


def test_coverage_reports_only_retained_notes_and_protects_confident_candidates():
    from audit_accompaniment_coverage import coverage

    item = fixture()
    item['keep'] = np.array([True, False, True])
    item['p'][2] = .9
    counts = coverage([item])
    assert counts == {'eligible': {'matched': 1, 'unmatched': 0},
                      'protected': {'matched': 0, 'unmatched': 1}}
