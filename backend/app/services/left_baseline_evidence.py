"""Label-free frozen-model confidence features shared by training and inference."""

import numpy as np

from app.services.candidate_relations import ALL_NAMES
from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES

EVIDENCE_VERSION = 'left-baseline-evidence-v1'
EVIDENCE_NAMES = ('v7_relation_confidence', 'v7_guardian_confidence')
RELATION_EVIDENCE_NAMES = ALL_NAMES + EVIDENCE_NAMES
ACOUSTIC_EVIDENCE_NAMES = FEATURE_NAMES + CONTEXT_NAMES + EVIDENCE_NAMES


def append_evidence(features, relation, guardian):
    features, relation, guardian = np.asarray(features), np.asarray(relation), np.asarray(guardian)
    if (features.ndim != 2 or features.shape[1] != 72 or not np.isfinite(features).all()
            or any(p.shape != (len(features),) or not np.isfinite(p).all()
                   or np.any((p < 0) | (p > 1)) for p in (relation, guardian))):
        raise ValueError('Invalid frozen left-hand evidence')
    return np.column_stack((features, relation, guardian)).astype(np.float32)


def acoustic_evidence(features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 74 or not np.isfinite(features).all():
        raise ValueError('Invalid acoustic left-hand evidence')
    return np.column_stack((features[:, :52], features[:, -2:]))
