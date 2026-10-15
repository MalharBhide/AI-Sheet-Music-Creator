"""Optional refinement that can only remove notes accepted by the approved model.

Routing remains on the approved verifier until a separately qualified asset is
explicitly configured. Both models observe the complete original candidate set;
the inherited proposal is still applied after full-context release correction.
"""

import numpy as np

from app.models import PipelineError
from app.services.left_presence import LeftHandPresence

APPROVED_SHA256 = "38899dd8dbbefafb3a2d5d2f210dc2e6db66087ffc6ee95fcae2635f5d9bbfda"


class IntersectedPresence:
    def __init__(self, approved, refinement):
        if any(not np.array_equal(a, b) or a.dtype != b.dtype
               for a, b in zip(approved.normalizer, refinement.normalizer, strict=True)):
            raise ValueError("Refinement must retain the approved normalization")
        self.approved, self.refinement = approved, refinement

    def keep(self, item, duration, stricter=False):
        parent = np.asarray(self.approved.keep(item, duration, stricter))
        extra = np.asarray(self.refinement.keep(item, duration, stricter))
        if (parent.shape != (len(item["events"]),) or extra.shape != parent.shape
                or parent.dtype != np.bool_ or extra.dtype != np.bool_):
            raise ValueError("Invalid intersected pitch-presence decisions")
        return parent & extra


class LeftHandPresenceRefinement(LeftHandPresence):
    name = "Left-hand pitch-presence refinement"

    def __init__(self, approved_asset, refinement_asset, refinement_sha):
        super().__init__(approved_asset, APPROVED_SHA256)
        refinement = LeftHandPresence(refinement_asset, refinement_sha).model
        try:
            self.model = IntersectedPresence(self.model, refinement)
        except ValueError as exc:
            raise PipelineError("The refinement differs from the approved model normalization.") from exc
