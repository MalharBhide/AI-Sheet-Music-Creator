"""Portable probability inference for locally trained accompaniment trees."""

import numpy as np

from app.services.note_evidence import FEATURE_NAMES, FEATURE_VERSION


class NoteForest:
    def __init__(self, saved):
        if (saved['version'] != 'note-forest-v1' or saved['feature_version'] != FEATURE_VERSION
                or saved['feature_names'] != list(FEATURE_NAMES)):
            raise ValueError('Incompatible accompaniment forest features')
        self.trees = []
        for tree in saved['trees']:
            arrays = {key: np.asarray(value) for key, value in tree.items()}
            size = len(arrays['left'])
            if any(value.shape != (size,) or not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError('Invalid tree arrays')
            left, right, feature = (arrays[key] for key in ('left', 'right', 'feature'))
            branches = left != -1
            indices = np.arange(size)
            # sklearn numbers children after parents; enforce an acyclic graph.
            if (not size or np.any((left[branches] <= indices[branches]) | (left[branches] >= size))
                    or np.any((right[branches] <= indices[branches]) | (right[branches] >= size))
                    or np.any((feature[branches] < 0) | (feature[branches] >= len(FEATURE_NAMES)))
                    or np.any(right[~branches] != -1)
                    or np.any((arrays['probability'] < 0) | (arrays['probability'] > 1))):
                raise ValueError('Invalid tree topology or probability')
            self.trees.append(arrays)
        if not self.trees:
            raise ValueError('Empty accompaniment forest')

    def probability(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('Invalid forest note features')
        total = np.zeros(len(x))
        for tree in self.trees:
            nodes = np.zeros(len(x), dtype=np.int64)
            while True:
                active = np.flatnonzero(tree['left'][nodes] != -1)
                if not len(active):
                    break
                current = nodes[active]
                left = x[active, tree['feature'][current]] <= tree['split'][current]
                nodes[active] = np.where(left, tree['left'][current], tree['right'][current])
            total += tree['probability'][nodes]
        return total / len(self.trees)
