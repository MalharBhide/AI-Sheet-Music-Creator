"""The portable runtime must preserve fitted tree predictions and reject cycles."""

import numpy as np
import pytest
from export_context_verifier import export
from sklearn.ensemble import HistGradientBoostingClassifier

from app.services.note_context_model import ContextNoteModel


def test_export_matches_sklearn_on_unseen_features_and_rejects_malformed_trees(tmp_path):
    rng = np.random.default_rng(82)
    x = rng.normal(size=(600, 52)).astype(np.float32)
    y = ((x[:, 0] > .2) | (x[:, 1] + x[:, 2] > .8)).astype(int)
    model = HistGradientBoostingClassifier(max_iter=12, max_leaf_nodes=7, min_samples_leaf=8,
                                           early_stopping=False, random_state=82).fit(x, y)
    unseen = rng.normal(size=(100, 52)).astype(np.float32)
    path = tmp_path / 'model.npz'
    export(model, .07, path, [{'x': unseen}])
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    runtime = ContextNoteModel(arrays)
    np.testing.assert_allclose(runtime.probability(unseen), model.predict_proba(unseen)[:, 1], atol=1e-12)
    assert runtime.probability(np.empty((0, 52))).shape == (0,)
    with pytest.raises(ValueError, match='input'):
        runtime.probability(np.full((1, 52), np.nan))
    arrays['left'][0, 0] = 0
    with pytest.raises(ValueError, match='topology'):
        ContextNoteModel(arrays)
