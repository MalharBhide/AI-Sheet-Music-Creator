"""Observed same-pitch release relationships, including long preceding holds."""

import numpy as np

from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES

NAMES = (
    "same_pitch_prior_release_strength",
    "same_pitch_prior_release_delta",
    "same_pitch_prior_duration",
    "same_pitch_prior_end_level",
    "same_pitch_prior_release_drop",
    "same_pitch_overlap_fraction",
    "same_pitch_overlap_strength",
    "same_pitch_next_attack_strength",
)
BASE_NAMES = FEATURE_NAMES + CONTEXT_NAMES


def features(events, base):
    events, base = np.asarray(events, float), np.asarray(base, np.float32)
    count = len(events)
    if (
        events.shape != (count, 4)
        or base.shape != (count, 52)
        or not np.isfinite(events).all()
        or not np.isfinite(base).all()
        or np.any(events[:, 0] < 0)
        or np.any(events[:, 1] <= events[:, 0])
    ):
        raise ValueError("Invalid observed release-neighbor clock or evidence")
    rows = []
    starts, ends, pitches, _ = events.T
    strength = base[:, BASE_NAMES.index("note_mean")]
    attacks = base[:, BASE_NAMES.index("onset_at_attack")]
    for index, (start, end, pitch, _) in enumerate(events):
        same = (pitches == pitch) & (np.arange(count) != index)
        prior = same & (starts < start) & (np.abs(ends - start) <= 0.25)
        overlaps = np.maximum(0, np.minimum(end, ends) - np.maximum(start, starts))
        if prior.any():
            indices = np.flatnonzero(prior)
            distances = np.abs(ends[indices] - start)
            nearest = indices[np.isclose(distances, distances.min(), rtol=0, atol=1e-9)]
            # Equal-time duplicates must not depend on input/instrument order.
            delta = float(np.clip(np.mean(ends[nearest] - start) / 0.25, -1, 1))
            duration = float(np.log1p(min(np.max(ends[nearest] - starts[nearest]), 8)) / np.log(9))
            level = float(np.mean(base[nearest, BASE_NAMES.index("note_end_level")]))
            drop = float(np.mean(base[nearest, BASE_NAMES.index("note_release_drop")]))
        else:
            delta, duration, level, drop = 0.0, 0.0, 0.0, 0.0
        near_next = same & (starts >= start) & (np.abs(starts - end) <= 0.25)
        fraction = overlaps / (end - start)
        rows.append(
            (
                float(
                    np.max(strength[prior] * np.exp(-np.abs(ends[prior] - start) / 0.08), initial=0)
                ),
                delta,
                duration,
                level,
                drop,
                float(np.max(fraction[same], initial=0)),
                float(np.max((strength * fraction)[same], initial=0)),
                float(
                    np.max(
                        attacks[near_next] * np.exp(-np.abs(starts[near_next] - end) / 0.08),
                        initial=0,
                    )
                ),
            )
        )
    result = np.asarray(rows, np.float32).reshape(-1, len(NAMES))
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise ValueError("Invalid bounded release-neighbor features")
    return result
