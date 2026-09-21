"""Export fitted sklearn trees as checked arrays, without executable pickle."""

import numpy as np

from app.services.note_context_model import ContextNoteModel
from app.services.note_evidence import CONTEXT_NAMES, CONTEXT_VERSION, FEATURE_NAMES


def export(model, threshold, path, validation):
    nodes = [predictors[0].nodes for predictors in model._predictors]
    shape = (len(nodes), max(map(len, nodes)))
    saved = {key: np.full(shape, -1, dtype=np.int64) for key in ('left', 'right', 'feature')}
    saved.update({key: np.zeros(shape, dtype=np.float64) for key in ('split', 'value')})
    for index, tree in enumerate(nodes):
        if tree['is_categorical'].any():
            raise ValueError('Categorical splits are not part of this model format')
        branch = np.flatnonzero(~tree['is_leaf'].astype(bool))
        for key, source in [('left', 'left'), ('right', 'right'), ('feature', 'feature_idx'), ('split', 'num_threshold')]:
            saved[key][index, branch] = tree[source][branch]
        saved['value'][index, :len(tree)] = tree['value']
    saved.update({'version': np.asarray('note-boosted-v1'), 'feature_version': np.asarray(CONTEXT_VERSION),
                  'feature_names': np.asarray(FEATURE_NAMES + CONTEXT_NAMES),
                  'intercept': np.asarray(float(model._baseline_prediction[0, 0])),
                  'threshold': np.asarray(threshold)})
    portable = ContextNoteModel(saved)
    for item in validation:
        np.testing.assert_allclose(portable.probability(item['x']), model.predict_proba(item['x'])[:, 1],
                                   rtol=1e-12, atol=1e-12)
    np.savez_compressed(path, **saved)
