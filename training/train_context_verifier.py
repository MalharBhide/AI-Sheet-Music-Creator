"""Train harmonic-context classifiers and select against the deployed verifier."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
import torch
from audit_verifier_labels import audit
from export_context_verifier import export
from note_verifier_model import NoteVerifier
from sklearn.ensemble import HistGradientBoostingClassifier
from train_note_verifier import aggregate, evaluate, load_data, metrics, write


def previous_metrics(directory, validation):
    old = torch.load(directory.parent / 'note-verifier-v2-conservative/candidate.pt',
                     weights_only=True, map_location='cpu')
    model = NoteVerifier().eval()
    model.load_state_dict(old['state_dict'])
    rows = []
    for item in validation:
        if 'legacy_x' in item:
            x, events = item['legacy_x'], item['legacy_events']
        else:
            with np.load(directory.parent / 'note-verifier-features' / (item['id'] + '.npz')) as cached:
                x, events = cached['x'], cached['events']
        keep = (events[:, 2] >= 36) & (events[:, 2] < 96)
        if 'evaluation_window' in item:
            start, stop = item['evaluation_window']
            keep &= (events[:, 0] >= start) & (events[:, 0] < stop)
        x, events = x[keep], events[keep]
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.from_numpy((x - old['mean'].numpy()) / old['scale'].numpy()))).numpy()
        rows.append(metrics(item['reference'], events[probability >= old['threshold']], item['seconds']))
    return {corpus: aggregate([row for item, row in zip(validation, rows, strict=True) if item['corpus'] == corpus])
            for corpus in sorted({item['corpus'] for item in validation})}


def train(directory, run_name):
    output = directory / run_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    write(output / 'label-audit.json', audit(training, validation))
    previous = previous_metrics(directory, validation)
    write(output / 'previous-validation.json', previous)
    all_x, all_y, weights = [], [], []
    for corpus in sorted({item['corpus'] for item in training}):
        records = [item for item in training if item['corpus'] == corpus]
        for item in records:
            x, y = item['x'][item['mask']], item['y'][item['mask']]
            all_x.append(x)
            all_y.append(y)
            weights.append(np.where(y == 1, 2., 1.) / (len(records) * len(y)))
    x, y, weight = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(weights)
    weight /= weight.mean()
    best = None
    for leaves in (15, 31):
        model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=leaves,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=1.,
            early_stopping=False, random_state=260921)
        model.fit(x, y, sample_weight=weight)
        probabilities = [model.predict_proba(item['x'])[:, 1] for item in validation]
        search = []
        for threshold in np.arange(0., .801, .01):
            result = evaluate(validation, probabilities, float(threshold))
            failed = [row['id'] for row in result['per_recording'] if
                      row['trained']['f1'] < row['baseline']['f1']
                      or row['trained']['recall'] < row['baseline']['recall'] - .01]
            regressions = [(corpus, metric) for corpus, row in result['aggregate'].items()
                           for metric in ('precision', 'recall', 'f1')
                           if row['trained'][metric] < previous[corpus][metric]]
            score = float(np.mean([row['trained']['f1'] for row in result['aggregate'].values()]))
            search.append({'threshold': float(threshold), 'recording_failures': failed,
                           'current_regressions': regressions, 'score': score, 'aggregate': result['aggregate']})
            if not failed and not regressions and (best is None or score > best[0]):
                best = (score, model, float(threshold), leaves, result)
        write(output / f'search-{leaves}.json', search)
        # Offline experiment only. Runtime export below contains arrays, no pickle.
        with (output / f'model-{leaves}.pickle').open('wb') as stream:
            pickle.dump(model, stream)
        print(json.dumps({'leaves': leaves, 'fewest_validation_failures': min(
            len(row['recording_failures']) + len(row['current_regressions']) for row in search)}), flush=True)
    write(output / 'run.json', {'seed': 260921, 'trees': 300, 'leaf_candidates': [15, 31],
        'minimum_leaf_events': 20, 'learning_rate': .05, 'l2': 1., 'positive_weight': 2.,
        'weighting': 'Equal corpora and equal recordings within corpus',
        'training_clips': len(training), 'training_events': len(x), 'validation_clips': len(validation),
        'sklearn_version': sklearn.__version__, 'test_used_for_selection': False})
    (output / 'split.json').write_bytes((directory / 'note-verifier-split.json').read_bytes())
    if best is None:
        write(output / 'selection.json', {'selected': False, 'reason': 'No candidate passed validation guards'})
        print('No eligible model; test evaluation is unnecessary.', flush=True)
        return
    _, model, threshold, leaves, result = best
    export(model, threshold, output / 'candidate.npz', validation)
    write(output / 'validation.json', result)
    write(output / 'selection.json', {'selected': True,
        'checkpoint_sha256': hashlib.sha256((output / 'candidate.npz').read_bytes()).hexdigest(),
        'threshold': threshold, 'leaves': leaves, 'test_not_evaluated': True,
        'selection': 'Maximum macro-corpus validation F1, nondecreasing per-recording F1 and <= .01 recall loss vs original detector, and nondecreasing precision/recall/F1 in each corpus vs deployed V2'})
    print(json.dumps({'selected_leaves': leaves, 'threshold': threshold}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--run', default='context-verifier')
    args = parser.parse_args()
    train(args.directory, args.run)
