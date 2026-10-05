"""Fixed-pair timing measurements: deleting notes earns no timing credit."""

import numpy as np
from bass_joint_timing import KEEP, actions, protected, transform
from train_melody_timing import matches
from train_note_verifier import aggregate, metrics


def timing(item, after, indices):
    old, reference = item['events'], item['reference']
    pairs = matches(reference, old, .2)
    # Deleted predictions retain their old error. Never rematch to an easier
    # note or shrink the denominator to make timing seem to improve.
    mapped = old.copy()
    mapped[indices] = after
    before, new = [], []
    by_pitch = {}
    for ref, index in pairs:
        before.append(np.abs(old[index, :2]-reference[ref, :2]))
        new.append(np.abs(mapped[index, :2]-reference[ref, :2]))
        by_pitch.setdefault(reference[ref, 2], []).append((ref, index))
    old_spacing, new_spacing = [], []
    for group in by_pitch.values():
        group.sort(key=lambda pair: reference[pair[0], 0])
        for (a, i), (b, j) in zip(group, group[1:], strict=False):
            target = reference[b, 0]-reference[a, 0]
            old_spacing.append(abs(old[j, 0]-old[i, 0]-target))
            new_spacing.append(abs(mapped[j, 0]-mapped[i, 0]-target))
    first = np.asarray(before).reshape(-1, 2).sum(axis=0)
    second = np.asarray(new).reshape(-1, 2).sum(axis=0)
    spacing_before, spacing_after = float(sum(old_spacing)), float(sum(new_spacing))
    return {'pairs': len(pairs), 'before_onset_release_error_sum': first.tolist(),
            'after_onset_release_error_sum': second.tolist(), 'repeated_spacing_pairs': len(old_spacing),
            'before_repeated_spacing_error_sum': spacing_before,
            'after_repeated_spacing_error_sum': spacing_after,
            'passes': bool(np.all(second <= first+1e-9) and spacing_after <= spacing_before+1e-9)}


def row(item, chosen, strength, memo):
    key = (item['id'], float(strength), chosen.astype(np.uint8).tobytes())
    if key in memo:
        return memo[key]
    new, indices = transform(item['events'], chosen, item['duration'], strength)
    safe, counts, coverage = protected(item, new)
    clock = timing(item, new, indices)
    baseline = metrics(item['reference'], item['events'], item['duration'])
    candidate = metrics(item['reference'], new, item['duration'])
    changed = int(np.sum(np.any(new[:, :2] != item['events'][indices, :2], axis=1)))
    result = {'id': item['id'], 'corpus': item['corpus'], 'baseline': baseline, 'candidate': candidate,
              'matches': counts, 'coverage': coverage, 'timing': clock, 'retimed_notes': changed,
              'removed_notes': len(item['events'])-len(new),
              'passes': bool(safe and clock['passes'] and candidate['false_positives'] <= baseline['false_positives'])}
    memo[key] = result
    return result


def evaluate(items, probabilities, guardians, timing_threshold, remove_threshold, strength=1., memo=None):
    memo = {} if memo is None else memo
    rows = [row(item, actions(item, p, g, timing_threshold, remove_threshold), strength, memo)
            for item, p, g in zip(items, probabilities, guardians, strict=True)]
    onset_gain = sum(r['timing']['before_onset_release_error_sum'][0]
                     -r['timing']['after_onset_release_error_sum'][0] for r in rows)
    groups = {}
    for corpus in sorted({r['corpus'] for r in rows}):
        group = [r for r in rows if r['corpus'] == corpus]
        pairs = sum(r['timing']['pairs'] for r in group)
        repeated = sum(r['timing']['repeated_spacing_pairs'] for r in group)
        groups[corpus] = {key: aggregate([r[key] for r in group]) for key in ('baseline', 'candidate')}
        groups[corpus]['timing'] = {'pairs': pairs, 'repeated_spacing_pairs': repeated}
        for system in ('before', 'after'):
            errors = np.sum([r['timing'][system+'_onset_release_error_sum'] for r in group], axis=0)
            groups[corpus]['timing'][system+'_onset_release_mae_seconds'] = (errors/max(1,pairs)).tolist()
            groups[corpus]['timing'][system+'_repeated_spacing_mae_seconds'] = (
                sum(r['timing'][system+'_repeated_spacing_error_sum'] for r in group)/max(1,repeated))
    return {'timing_threshold': timing_threshold, 'remove_threshold': remove_threshold, 'strength': strength,
            'passes': all(r['passes'] for r in rows), 'failed_recordings': [r['id'] for r in rows if not r['passes']],
            'onset_error_reduction_seconds': float(onset_gain), 'retimed_notes': sum(r['retimed_notes'] for r in rows),
            'false_notes_removed': sum(r['baseline']['false_positives']-r['candidate']['false_positives'] for r in rows),
            'aggregate': groups, 'per_recording': rows}


def choose(items, probabilities, guardians, thresholds, memo):
    eligible, search = [], []
    for timing_threshold in thresholds:
        for remove_threshold in thresholds:
            raw = evaluate(items, probabilities, guardians, timing_threshold, remove_threshold, memo=memo)
            margin = evaluate(items, probabilities, guardians, min(1.01,timing_threshold+.05),
                              min(1.01,remove_threshold+.05), .5, memo)
            passes = (raw['passes'] and margin['passes']
                      and raw['onset_error_reduction_seconds'] > 1e-6
                      and margin['onset_error_reduction_seconds'] > 1e-6)
            summary = {k:raw[k] for k in ('timing_threshold','remove_threshold','passes','failed_recordings',
                       'retimed_notes','false_notes_removed','onset_error_reduction_seconds')}
            summary.update(eligible=bool(passes), margin_passes=margin['passes'],
                           margin_failed_recordings=margin['failed_recordings'],
                           margin_onset_error_reduction_seconds=margin['onset_error_reduction_seconds'])
            search.append(summary)
            if passes:
                key = (raw['onset_error_reduction_seconds'], raw['false_notes_removed'], timing_threshold, remove_threshold)
                eligible.append((key, {'raw':raw,'margin':margin}))
    winner = max(eligible, key=lambda pair: pair[0])[1] if eligible else None
    return winner, search


def baseline_actions(item):
    return np.full(len(item['events']), KEEP, np.int64)
