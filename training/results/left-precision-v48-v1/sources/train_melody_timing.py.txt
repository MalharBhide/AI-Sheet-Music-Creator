"""Fit note-boundary corrections from labeled cached features, never user audio.

The existing melody pitches/counts are frozen. A candidate may only retime notes
within 80 ms, preserving order, positive duration and monophony. Validation and
regression protect both onset and onset/offset reference matches.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from train_context_melody import baseline_model, load_vocalset
from train_melody import load_data, predict
from train_note_verifier import write

from app.services.melody_decoder import RATE, decode
from app.services.vocal_melody import CHECKPOINT


def boundary_features(x, events, side):
    rows = []
    for start, end, pitch in events:
        center = round((start if side == 0 else end) * RATE)
        frames = np.clip(center + np.arange(-6, 7), 0, len(x) - 1)
        evidence = x[frames, int(pitch) - 21].reshape(-1)
        rows.append(np.r_[evidence, min(4., end - start) / 4])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 131)


def matches(reference, events, tolerance=.05, offsets=False):
    return mir_eval.transcription.match_notes(reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]), onset_tolerance=tolerance,
        pitch_tolerance=50, offset_ratio=.2 if offsets else None, offset_min_tolerance=.05)


def retime(events, corrections, scale, duration):
    result = events.copy()
    if not len(events) or duration < .02:
        return result
    result[:, :2] += np.clip(corrections, -.08, .08) * scale
    result[:, 0] = np.clip(result[:, 0], 0., duration - .02)
    result[:, 1] = np.clip(result[:, 1], .02, duration)
    # Freeze the note ordering: adjacent boundaries cannot cross. If two
    # proposals overlap, meet at their midpoint without creating an extra note.
    for index in range(len(result) - 1):
        if result[index, 1] > result[index + 1, 0]:
            middle = (result[index, 1] + result[index + 1, 0]) / 2
            result[index, 1] = middle
            result[index + 1, 0] = middle
    invalid = result[:, 1] - result[:, 0] < .02
    if invalid.any():
        # Protect the entire phrase if its segmentation would be broken.
        return events.copy()
    return result


def dataset(directory, group, output):
    split = json.loads((directory / 'melody-v1/split.json').read_text())
    collections = [('vocadito', load_data(directory, split['tracks'][group])),
                   ('vocalset', load_vocalset(directory, group))]
    model, checkpoint = baseline_model()
    cache = output / 'cached-boundaries'
    cache.mkdir(exist_ok=True)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    result = []
    for corpus, items in collections:
        for item in items:
            identity = corpus + '-' + str(item['id'])
            path = cache / (identity + '.npz')
            if not path.exists():
                logits, attacks = predict(model, [item])[0]
                events = decode(logits, attacks, **checkpoint['decoder'])
                refs = np.array([[s, s + length, float(librosa.hz_to_midi(hz))]
                                 for s, hz, length in item['A1']]).reshape(-1, 3)
                np.savez_compressed(path, events=events, reference=refs,
                    onset_x=boundary_features(item['x'], events, 0),
                    release_x=boundary_features(item['x'], events, 1),
                    duration=len(item['x']) / RATE, model_sha256=digest)
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['model_sha256']) != digest:
                    raise ValueError('Stale timing feature cache')
                row = {k: saved[k] for k in ('events', 'reference', 'onset_x', 'release_x', 'duration')}
            result.append({**row, 'id': identity, 'corpus': corpus})
        print(json.dumps({'cached': group, 'corpus': corpus, 'clips': len(items)}), flush=True)
    return result


def evaluate(items, models, scale):
    rows = []
    for item in items:
        old = item['events']
        shifts = np.column_stack([model.predict(item[key]) for model, key in
                                   zip(models, ('onset_x', 'release_x'), strict=True)]) if len(old) else np.empty((0, 2))
        new = retime(old, shifts, scale, float(item['duration']))
        loose = matches(item['reference'], old, .2)
        old_errors, new_errors = [], []
        for i, j in loose:
            old_errors.append(np.abs(old[j, :2] - item['reference'][i, :2]))
            new_errors.append(np.abs(new[j, :2] - item['reference'][i, :2]))
        preserved, counts = True, {}
        for name, offsets in [('onset', False), ('onset_offset', True)]:
            a = {i for i, _ in matches(item['reference'], old, offsets=offsets)}
            b = {i for i, _ in matches(item['reference'], new, offsets=offsets)}
            preserved &= a.issubset(b)
            counts[name] = {'before': len(a), 'after': len(b), 'lost': len(a - b)}
        before = np.asarray(old_errors).reshape(-1, 2)
        after = np.asarray(new_errors).reshape(-1, 2)
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'matches': counts,
            'timing_pairs': len(loose), 'before_error_sum': before.sum(axis=0).tolist(),
            'after_error_sum': after.sum(axis=0).tolist(), 'passes': bool(preserved)})
    totals = {}
    for corpus in sorted({item['corpus'] for item in items}):
        group = [row for row in rows if row['corpus'] == corpus]
        count = sum(row['timing_pairs'] for row in group)
        totals[corpus] = {key: (np.sum([row[key] for row in group], axis=0) / max(1, count)).tolist()
                         for key in ('before_error_sum', 'after_error_sum')}
        totals[corpus]['timing_pairs'] = count
    return {'scale': scale, 'passes': all(row['passes'] for row in rows),
            'failed_recordings': [row['id'] for row in rows if not row['passes']],
            'mean_absolute_onset_release_error_seconds': totals, 'per_recording': rows}


def train(directory, output):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = [dataset(directory, group, output) for group in ('train', 'validation')]
    settings = dict(loss='absolute_error', max_iter=150, max_leaf_nodes=7,
                    min_samples_leaf=40, l2_regularization=4., learning_rate=.05,
                    early_stopping=False, random_state=260924)
    write(output / 'run.json', {'settings': settings, 'train_clips': len(training),
        'validation_clips': len(validation), 'max_shift_seconds': .08,
        'scales': [.25, .5, 1.], 'baseline_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'sklearn_version': sklearn.__version__, 'scope': 'Cached licensed singing features; fixed pitch/counts',
        'matching': 'Training uses same-pitch one-to-one matches within 200 ms; gates use 50 ms onset and 20%/50 ms offset',
        'weighting': 'Equal weight per corpus and per recording before event weighting'})
    models, training_counts = [], []
    for side, key in enumerate(('onset_x', 'release_x')):
        xs, ys, weights = [], [], []
        for item in training:
            pairs = matches(item['reference'], item['events'], .2)
            if not pairs:
                continue
            a, b = np.asarray(pairs).T
            target = item['reference'][a, side] - item['events'][b, side]
            usable = np.abs(target) <= .2
            if not usable.any():
                continue
            xs.append(item[key][b[usable]])
            ys.append(target[usable])
            corpus_count = sum(i['corpus'] == item['corpus'] for i in training)
            weights.append(np.full(usable.sum(), 1 / (corpus_count * usable.sum())))
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
        model = HistGradientBoostingRegressor(**settings).fit(x, y, sample_weight=w / w.mean())
        models.append(model)
        training_counts.append(len(y))
    with (output / 'candidate.pickle').open('wb') as stream:
        pickle.dump(models, stream)
    reports = [evaluate(validation, models, scale) for scale in (.25, .5, 1.)]
    write(output / 'validation.json', reports)
    eligible = [r for r in reports if r['passes'] and all(
        np.all(np.asarray(v['after_error_sum']) <= np.asarray(v['before_error_sum']))
        for v in r['mean_absolute_onset_release_error_seconds'].values())]
    selected = min(eligible, key=lambda r: sum(sum(v['after_error_sum']) for v in
                   r['mean_absolute_onset_release_error_seconds'].values())) if eligible else None
    result = {'selected': selected is not None, 'scale': selected['scale'] if selected else None,
        'training_events_per_head': training_counts,
        'candidate_sha256': hashlib.sha256((output / 'candidate.pickle').read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'selection': 'No lost onset or onset/offset reference matches in any clip; no corpus onset/release MAE regression',
        'test_evaluated': False}
    write(output / 'selection.json', result)
    print(json.dumps({**result, 'validation': [{k: v for k, v in r.items() if k != 'per_recording'} for r in reports]}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='melody-timing-v1')
    args = parser.parse_args()
    train(args.directory, args.directory / args.run)
