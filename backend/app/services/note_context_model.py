"""Data-only inference for a locally trained accompaniment context classifier."""

import numpy as np

from app.services.note_evidence import CONTEXT_NAMES, CONTEXT_VERSION, FEATURE_NAMES


def shared_events(events, bounded_events):
    """Only change candidates reproduced exactly by the training decoder."""
    keys = {tuple(event) for event in bounded_events}
    return np.asarray([tuple(event) in keys for event in events], dtype=bool)


def correction_keep(previous, context, shared, *, threshold=.1, ceiling=.2, prune=.05):
    """Preserve confident V2 events; require both heads to reject an uncertain one."""
    previous, context, shared = np.asarray(previous), np.asarray(context), np.asarray(shared)
    if (previous.ndim != 1 or context.shape != (2, len(previous)) or shared.shape != previous.shape
            or shared.dtype.kind != 'b' or not np.isfinite(previous).all()
            or not np.isfinite(context).all() or np.any((previous < 0) | (previous > 1))
            or np.any((context < 0) | (context > 1)) or not 0 <= threshold <= ceiling <= 1
            or not 0 <= prune <= 1):
        raise ValueError('Invalid accompaniment correction confidence')
    rejected = shared & (previous <= ceiling) & np.all(context < prune, axis=0)
    return (previous >= threshold) & ~rejected


class ContextNoteModel:
    def __init__(self, saved):
        if (str(saved['version']) != 'note-boosted-v1'
                or str(saved['feature_version']) != CONTEXT_VERSION
                or list(saved['feature_names']) != list(FEATURE_NAMES + CONTEXT_NAMES)):
            raise ValueError('Incompatible accompaniment context features')
        self.threshold = float(saved['threshold'])
        self.intercept = float(saved['intercept'])
        self.left, self.right, self.feature = (np.asarray(saved[k]) for k in ('left', 'right', 'feature'))
        self.split, self.value = (np.asarray(saved[k]) for k in ('split', 'value'))
        shape = self.left.shape
        arrays = (self.left, self.right, self.feature, self.split, self.value)
        if (len(shape) != 2 or not 0 < shape[0] <= 1024 or not 0 < shape[1] <= 255
                or any(a.shape != shape or not np.isfinite(a).all() for a in arrays)
                or any(a.dtype.kind not in 'iu' for a in (self.left, self.right, self.feature))
                or not np.isfinite(self.intercept) or not 0 <= self.threshold <= 1):
            raise ValueError('Invalid accompaniment context model arrays')
        parent = np.broadcast_to(np.arange(shape[1]), shape)
        branch = self.left != -1
        if (np.any((self.left[branch] <= parent[branch]) | (self.left[branch] >= shape[1]))
                or np.any((self.right[branch] <= parent[branch]) | (self.right[branch] >= shape[1]))
                or np.any(self.right[~branch] != -1)
                or np.any((self.feature[branch] < 0) | (self.feature[branch] >= len(FEATURE_NAMES + CONTEXT_NAMES)))):
            raise ValueError('Invalid accompaniment context tree topology')

    def probability(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES + CONTEXT_NAMES) or not np.isfinite(x).all():
            raise ValueError('Invalid accompaniment context input')
        logits = np.full(len(x), self.intercept, dtype=np.float64)
        for left, right, feature, split, value in zip(
                self.left, self.right, self.feature, self.split, self.value, strict=True):
            nodes = np.zeros(len(x), dtype=np.int64)
            while True:
                active = np.flatnonzero(left[nodes] != -1)
                if not len(active):
                    break
                current = nodes[active]
                take_left = x[active, feature[current]] <= split[current]
                nodes[active] = np.where(take_left, left[current], right[current])
            logits += value[nodes]
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
