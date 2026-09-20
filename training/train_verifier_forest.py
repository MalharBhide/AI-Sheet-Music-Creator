"""Fit a conservative tree ensemble on repaired training labels.

Use only training events for fitting and validation recordings for threshold
selection. Previously consumed evaluation recordings remain regression checks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from audit_verifier_labels import audit
from sklearn.ensemble import ExtraTreesClassifier
from train_note_verifier import evaluate, load_data, write

from app.services.note_evidence import FEATURE_NAMES, FEATURE_VERSION


def train(directory):
    output = directory / 'note-verifier-v4'
    output.mkdir(exist_ok=False)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    write(output / 'label-audit.json', audit(training, validation))
    x, y, weights = [], [], []
    corpora = {item['corpus'] for item in training}
    for corpus in sorted(corpora):
        records = [item for item in training if item['corpus'] == corpus]
        for item in records:
            features, labels = item['x'][item['mask']], item['y'][item['mask']]
            x.append(features)
            y.append(labels)
            # Equal recording weighting within each equally weighted corpus.
            weights.append(np.where(labels == 1, 4., 1.) / (len(records) * len(labels)))
    model = ExtraTreesClassifier(n_estimators=128, max_depth=12, min_samples_leaf=8,
                                 max_features=.8, random_state=260920, n_jobs=2)
    model.fit(np.concatenate(x), np.concatenate(y), sample_weight=np.concatenate(weights))
    predictions = [model.predict_proba(item['x'])[:, 1] for item in validation]
    best, chosen, selection = -1., 0., None
    search = []
    for threshold in np.arange(0., .901, .01):
        result = evaluate(validation, predictions, float(threshold))
        eligible = all(row['trained']['f1'] >= row['baseline']['f1']
                       and row['trained']['recall'] >= row['baseline']['recall'] - .01
                       for row in result['per_recording']) and all(
                           row['trained']['recall'] >= row['baseline']['recall'] - .0025
                           for row in result['aggregate'].values())
        score = float(np.mean([row['trained']['f1'] for row in result['aggregate'].values()]))
        search.append({'threshold': float(threshold), 'eligible': eligible, 'aggregate': result['aggregate']})
        if eligible and score > best:
            best, chosen, selection = score, float(threshold), result
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
                      'feature': tree.feature.tolist(), 'split': tree.threshold.tolist(),
                      'probability': (values[:, 1] / values.sum(axis=1)).tolist()})
    checkpoint = {'version': 'note-forest-v1', 'feature_version': FEATURE_VERSION,
                  'feature_names': list(FEATURE_NAMES), 'threshold': chosen, 'trees': trees}
    write(output / 'candidate.json', checkpoint)
    # Retain exact sklearn probabilities to verify the portable evaluator later.
    np.savez_compressed(output / 'export-check.npz', x=validation[0]['x'], probability=predictions[0])
    digest = hashlib.sha256((output / 'candidate.json').read_bytes()).hexdigest()
    write(output / 'selection.json', {'checkpoint_sha256': digest, 'threshold': chosen,
                                      'selection': 'Maximum macro-corpus F1 under per-recording F1/recall and corpus recall guards; validation only',
                                      'test_not_evaluated': True})
    write(output / 'validation.json', selection)
    write(output / 'threshold-search.json', search)
    write(output / 'run.json', {'training_clips': len(training), 'training_candidates': sum(map(len, y)),
                              'validation_clips': len(validation), 'trees': 128, 'max_depth': 12,
                              'minimum_leaf_events': 8, 'max_features': .8, 'seed': 260920,
                              'sklearn_version': sklearn.__version__,
                              'positive_weight': 4, 'sampling': 'Equal corpus and recording weights',
                              'trainable': 'ExtraTrees splits and leaf probabilities; frozen Basic Pitch'})
    print(json.dumps({'sha256': digest, 'threshold': chosen, 'validation': selection['aggregate']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    train(parser.parse_args().directory)
