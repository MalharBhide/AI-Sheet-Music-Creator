"""Shared production relationship features; retained imports for older experiments."""

from app.services.candidate_relations import (
    ALL_NAMES,
    CONTEXT_SECONDS,
    EDGE_SECONDS,
    RELATION_NAMES,
    RELATION_VERSION,
    complete_context,
    relation_features,
    window_eligible,
)

__all__ = [
    'ALL_NAMES', 'CONTEXT_SECONDS', 'EDGE_SECONDS', 'RELATION_NAMES',
    'RELATION_VERSION', 'complete_context', 'relation_features', 'window_eligible',
]
