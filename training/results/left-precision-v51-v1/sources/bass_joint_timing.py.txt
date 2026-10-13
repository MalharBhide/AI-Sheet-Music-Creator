"""Bounded raw-bass attack corrections and unsupported-note rejection."""

import numpy as np
from app.services.temporal_note_model import create_model
from bass_temporal_data import annotate
from pitch_interval_coverage import compare
from train_melody_timing import matches

VERSION = 'bass-v16-joint-attack-correction-v1'
EARLY, KEEP, LATE, REMOVE = range(4)
SHIFTS = np.array([-.02, 0., .02, 0.])


def model(width=48):
    import torch
    from torch import nn

    result = create_model(width)
    result.classifier[-1] = nn.Linear(32, 4)
    nn.init.zeros_(result.classifier[-1].weight)
    with torch.no_grad():
        result.classifier[-1].bias.copy_(torch.tensor([-3., 3., -3., -3.]))
    return result


def probabilities(network, frames, context, normalizer, batch_size=256):
    import torch

    mean, scale = normalizer
    frames, context = np.asarray(frames, np.float32), np.asarray(context, np.float32)
    if (context.shape != (len(frames), 84) or frames.shape != (len(context), 40, 9)
            or any(not np.isfinite(a).all() for a in (frames, context, mean, scale))
            or np.asarray(mean).shape != (84,) or np.asarray(scale).shape != (84,) or np.any(scale <= 0)
            or batch_size < 1 or np.any((frames < 0) | (frames > 1))):
        raise ValueError('Invalid joint timing model evidence')
    network.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            x = torch.from_numpy(((context[start:start+batch_size]-mean)/scale).astype(np.float32))
            f = torch.from_numpy(frames[start:start+batch_size])
            output.append(torch.softmax(network(f, x), dim=-1).numpy())
    return np.concatenate(output) if output else np.empty((0, 4), np.float32)


def transform(events, action, duration, strength=1.):
    """Preserve retained pitch/velocity; touching same-key releases move together."""
    events, action = np.asarray(events, float), np.asarray(action)
    n = len(events)
    if (events.shape != (n, 4) or action.shape != (n,) or not np.isfinite(events).all()
            or np.any(~np.isin(action, [EARLY, KEEP, LATE, REMOVE])) or strength not in (.5, 1.)
            or not np.isfinite(duration) or duration <= 0
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > duration)):
        raise ValueError('Invalid joint boundary actions')
    original_eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration-2.5)
                         & (events[:, 2] >= 21) & (events[:, 2] < 60))
    action = np.where(original_eligible, action, KEEP)
    keep = action != REMOVE
    result = events.copy()
    by_pitch = {}
    for index in np.argsort(events[:, 0], kind='stable'):
        if keep[index]:
            by_pitch.setdefault(events[index, 2], []).append(index)
    for indices in by_pitch.values():
        for slot, index in enumerate(indices):
            if action[index] not in (EARLY, LATE):
                continue
            start = events[index, 0] + SHIFTS[action[index]]*strength
            previous = indices[slot-1] if slot else None
            following = indices[slot+1] if slot+1 < len(indices) else None
            if (start < 2.5 or result[index, 1]-start < .045
                    or (previous is not None and start-result[previous, 0] < .045)
                    or (following is not None and events[following, 0]-start < .045)):
                continue
            if previous is not None:
                touching = abs(events[previous, 1]-events[index, 0]) <= 1e-6
                if touching:
                    result[previous, 1] = start
                elif result[previous, 1] <= events[index, 0] and result[previous, 1] > start:
                    continue  # A real same-key rest cannot become a new overlap.
            result[index, 0] = start
    changed = result[keep]
    if (np.any(changed[:, 1] <= changed[:, 0])
            or np.any(np.abs(result[:, :2]-events[:, :2]) > .020001)
            or np.any(result[keep, 2:] != events[keep, 2:])):
        raise ValueError('Invalid joint correction or changed pitch/velocity')
    return changed, np.flatnonzero(keep)


def protected(item, after):
    before, reference = item['events'], item['reference']
    counts = {}
    passes = True
    for name, offsets in (('attack', False), ('hold', True)):
        old = {i for i, _ in matches(reference, before, offsets=offsets)}
        new = {i for i, _ in matches(reference, after, offsets=offsets)}
        counts[name] = {'before': len(old), 'after': len(new), 'lost': len(old-new)}
        passes &= old.issubset(new)
    coverage = compare(item['pitch_reference'], before, after)
    return bool(passes and coverage['passes']), counts, coverage


def targets(item):
    labeled = annotate(item)
    y = np.full(len(item['events']), KEEP, dtype=np.int64)
    y[(labeled['y'] == 0) & labeled['mask']] = REMOVE
    for ref, index in matches(item['reference'], item['events'], .2):
        if not item['eligible'][index] or labeled['y'][index] != 1 or not labeled['mask'][index]:
            continue
        delta = item['reference'][ref, 0]-item['events'][index, 0]
        if abs(delta) < .03:
            continue
        candidate = np.full(len(y), KEEP)
        candidate[index] = EARLY if delta < 0 else LATE
        new, _ = transform(item['events'], candidate, item['duration'])
        if np.array_equal(new[index, :2], item['events'][index, :2]):
            continue
        if protected(item, new)[0]:
            y[index] = candidate[index]
    return {**labeled, 'action_y': y}


def actions(item, probability, guardian, timing_threshold, remove_threshold):
    p, g = np.asarray(probability), np.asarray(guardian)
    if (p.shape != (len(item['events']), 4) or g.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(g).all()
            or np.any((p < 0) | (p > 1)) or np.any((g < 0) | (g > 1))
            or not np.allclose(p.sum(axis=1), 1., atol=1e-5)
            or not np.isfinite([timing_threshold, remove_threshold]).all()
            or not 0 <= timing_threshold <= 1.01 or not 0 <= remove_threshold <= 1.01):
        raise ValueError('Invalid joint action probabilities')
    chosen = p.argmax(axis=1)
    confident = p.max(axis=1)
    timing = ((chosen == EARLY) | (chosen == LATE)) & (confident >= timing_threshold)
    remove = (chosen == REMOVE) & (confident >= remove_threshold) & (g < .3)
    return np.where(item['eligible'] & (timing | remove), chosen, KEEP)
