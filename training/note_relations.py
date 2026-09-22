"""Pitch-relative candidate relationships for an offline note-filter experiment.

Only predicted notes and frozen V2 confidence are used, never reference labels.
One bounded inference window is processed at a time, without an N-by-N matrix.
"""

import numpy as np

from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES

RELATION_VERSION = 'note-relations-bounded-v2'
CONTEXT_SECONDS = 2.
RELATION_NAMES = (
    'v2_confidence', 'coattack_count', 'overlap_count', 'overlap_confidence_rank',
    'relative_velocity', 'coattack_confidence', 'release_alignment', 'relative_duration',
    'octave_coattack', 'nineteen_coattack', 'two_octaves_coattack',
    'octave_overlap', 'nineteen_overlap', 'two_octaves_overlap',
    'same_pitch_previous', 'same_pitch_next', 'near_pitch_previous', 'near_pitch_next',
    'previous_attack_proximity', 'next_attack_proximity',
)
ALL_NAMES = FEATURE_NAMES + CONTEXT_NAMES + RELATION_NAMES


def complete_context(events, start, stop):
    """Only score targets whose entire neighborhood exists in the observed crop."""
    events = np.asarray(events)
    return (events[:, 0] > start + CONTEXT_SECONDS) & (events[:, 0] < stop - CONTEXT_SECONDS)


def relation_features(events, acoustic, confidence):
    events = np.asarray(events, dtype=float)
    acoustic = np.asarray(acoustic, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=float)
    count = len(events)
    if (events.shape != (count, 4) or acoustic.shape != (count, 52)
            or confidence.shape != (count,) or not np.isfinite(events).all()
            or not np.isfinite(acoustic).all() or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))
            or np.any(events[:, 1] <= events[:, 0])):
        raise ValueError('Invalid candidate relationships')
    if not count:
        return np.empty((0, len(ALL_NAMES)), dtype=np.float32)
    starts, ends, pitches, velocities = events.T
    durations = ends - starts
    result = []
    for index, (start, end, pitch, velocity) in enumerate(events):
        delta = starts - start
        others = (np.arange(count) != index) & (np.abs(delta) <= CONTEXT_SECONDS)
        coattack = others & (np.abs(delta) <= .06)
        overlap = np.maximum(0., np.minimum(end, ends) - np.maximum(start, starts))
        active = others & (overlap > 0)

        def strength(mask, factor=None):
            weights = confidence if factor is None else confidence * factor
            return float(np.max(weights[mask], initial=0.))

        row = [confidence[index], np.log1p(coattack.sum()) / np.log(13),
               np.log1p(active.sum()) / np.log(25),
               float(np.mean(confidence[active] <= confidence[index])) if active.any() else 1.,
               velocity / max(1., float(np.max(velocities[active], initial=velocity))),
               strength(coattack), strength(others & (np.abs(ends - end) <= .06)),
               float(np.clip(durations[index] / max(.01, np.median(durations[active])), 0, 4) / 4)
               if active.any() else .25]
        row.extend(strength(coattack & (pitches == pitch - interval)) for interval in (12, 19, 24))
        row.extend(strength(active & (pitches == pitch - interval), overlap / durations[index])
                   for interval in (12, 19, 24))
        for nearby in ((pitches == pitch), (np.abs(pitches - pitch) <= 2)):
            for direction in (-1, 1):
                mask = others & nearby & (delta * direction > .06)
                row.append(strength(mask, np.exp(-np.abs(delta))))
        for direction in (-1, 1):
            distances = np.abs(delta[others & (delta * direction > .06)])
            row.append(float(np.exp(-distances.min())) if len(distances) else 0.)
        result.append(row)
    return np.concatenate([acoustic, np.asarray(result, dtype=np.float32)], axis=1)
