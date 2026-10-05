"""Checked sklearn inference for a bounded stem, including no-note windows."""

import numpy as np


def probability(model, x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model.n_features_in_ or not np.isfinite(x).all():
        raise ValueError('Invalid bass classifier feature contract')
    if not len(x):
        return np.empty(0, dtype=np.float64)
    return model.predict_proba(x)[:, 1]
