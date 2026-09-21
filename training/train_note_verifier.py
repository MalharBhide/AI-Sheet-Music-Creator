"""Train a note verifier; freeze validation selection before a one-shot test."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import torch
from audit_verifier_labels import audit
from note_verifier_model import NoteVerifier
from torch.nn import functional as F

from app.services.note_evidence import CONTEXT_VERSION, FEATURE_NAMES, FEATURE_VERSION

SEED = 260920


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_data(directory, group):
    manifest = json.loads((directory / 'note-verifier-split.json').read_text())
    items = []
    for entry in manifest['tracks'][group]:
        with np.load(directory / 'note-verifier-features' / (entry['id'] + '.npz')) as saved:
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            version = CONTEXT_VERSION if entry.get('context_features') else FEATURE_VERSION
            if str(saved['version']) != version or str(saved['identity']) != digest:
                raise ValueError('Stale feature cache')
            fields = ['x', 'y', 'mask', 'events', 'reference', 'seconds']
            if 'legacy_x' in saved:
                fields.extend(['legacy_x', 'legacy_events'])
            items.append({**entry, **{k: saved[k] for k in fields}})
    return items


def metrics(reference, events, seconds):
    matched = mir_eval.transcription.match_notes(reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]), onset_tolerance=.05, offset_ratio=None)
    return counts(len(matched), len(events), len(reference), float(seconds))


def counts(tp, predicted, reference, seconds):
    precision = tp / max(1, predicted)
    recall = tp / max(1, reference)
    return {'precision': precision, 'recall': recall, 'f1': 2 * tp / max(1, predicted + reference),
            'true_positives': tp, 'false_positives': predicted - tp, 'predicted': predicted,
            'reference': reference, 'seconds': seconds,
            'false_positives_per_minute': (predicted - tp) * 60 / max(1, seconds)}


def aggregate(rows):
    return counts(*(sum(row[k] for row in rows) for k in ('true_positives', 'predicted', 'reference', 'seconds')))


def probabilities(model, items, mean, scale):
    with torch.inference_mode():
        return [torch.sigmoid(model(torch.from_numpy((item['x'] - mean) / scale))).numpy() for item in items]


def evaluate(items, probabilities, threshold):
    rows = []
    for item, probability in zip(items, probabilities, strict=True):
        baseline = metrics(item['reference'], item['events'], item['seconds'])
        trained = metrics(item['reference'], item['events'][probability >= threshold], item['seconds'])
        rows.append({'id': item['id'], 'corpus': item['corpus'], 'baseline': baseline, 'trained': trained})
    result = {'per_recording': rows, 'aggregate': {}}
    for corpus in sorted({item['corpus'] for item in items}):
        result['aggregate'][corpus] = {
            system: aggregate([r[system] for r in rows if r['corpus'] == corpus])
            for system in ('baseline', 'trained')}
    return result


def passes_guard(result):
    return all(row['trained']['recall'] >= row['baseline']['recall'] - .02
               and row['trained']['f1'] >= row['baseline']['f1']
               for row in result['aggregate'].values())


def train(directory, epochs, run_name='note-verifier-v1', recall_margin=.01,
          positive_weight=2., protect_recordings=False):
    output = directory / run_name
    if output.exists():
        raise ValueError('Preserve earlier attempts: output directory already exists')
    output.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    training, validation = load_data(directory, 'train'), load_data(directory, 'validation')
    write(output / 'label-audit.json', audit(training, validation))
    pools = []
    for corpus in sorted({item['corpus'] for item in training}):
        items = [item for item in training if item['corpus'] == corpus]
        pools.append((np.concatenate([i['x'][i['mask']] for i in items]),
                      np.concatenate([i['y'][i['mask']] for i in items])))
    all_x = np.concatenate([x for x, _ in pools])
    mean, scale = all_x.mean(axis=0), np.maximum(.03, all_x.std(axis=0))
    pools = [(torch.from_numpy((x - mean) / scale), torch.from_numpy(y)) for x, y in pools]
    val_pools = []
    for corpus in sorted({item['corpus'] for item in training}):
        items = [i for i in validation if i['corpus'] == corpus]
        val_pools.append((torch.from_numpy((np.concatenate([i['x'][i['mask']] for i in items]) - mean) / scale),
                          torch.from_numpy(np.concatenate([i['y'][i['mask']] for i in items]))))
    model = NoteVerifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.002)
    history, best = [], float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(80):
            # Equal corpus weighting; sample events only from the training split.
            batches = [(x[index], y[index]) for x, y in pools
                       for index in [rng.integers(len(x), size=256)]]
            x, y = (torch.cat([batch[k] for batch in batches]) for k in (0, 1))
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(positive_weight))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(torch.stack([F.binary_cross_entropy_with_logits(model(x), y,
                pos_weight=torch.tensor(positive_weight)) for x, y in val_pools]).mean())
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_loss': val_loss})
        if val_loss < best:
            best, state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch + 1
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(state)
    predictions = probabilities(model, validation, mean, scale)
    # Threshold is selected on validation only, separately protecting each corpus.
    chosen, score, selection = 0., -float('inf'), None
    thresholds = []
    for threshold in np.arange(0., .801, .02):
        result = evaluate(validation, predictions, float(threshold))
        eligible = passes_guard(result) and all(row['trained']['recall'] >= row['baseline']['recall'] - recall_margin for row in result['aggregate'].values())
        if protect_recordings:
            eligible = eligible and all(
                row['trained']['f1'] >= row['baseline']['f1']
                and row['trained']['recall'] >= row['baseline']['recall'] - .01
                for row in result['per_recording'])
        f1 = float(np.mean([r['trained']['f1'] for r in result['aggregate'].values()]))
        thresholds.append({'threshold': float(threshold), 'eligible': eligible,
                           'aggregate': result['aggregate']})
        if eligible and f1 > score:
            chosen, score, selection = float(threshold), f1, result
    checkpoint = {'state_dict': state, 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                  'threshold': chosen, 'feature_version': FEATURE_VERSION, 'feature_names': list(FEATURE_NAMES)}
    torch.save(checkpoint, output / 'candidate.pt')
    digest = hashlib.sha256((output / 'candidate.pt').read_bytes()).hexdigest()
    write(output / 'history.json', history)
    write(output / 'validation.json', selection)
    write(output / 'threshold-search.json', thresholds)
    write(output / 'selection.json', {'checkpoint_sha256': digest, 'threshold': chosen,
        'epoch': best_epoch, 'validation_recall_margin': recall_margin,
        'protect_each_recording': protect_recordings,
        'selection': 'Minimum corpus-balanced validation BCE; then maximum macro-corpus validation F1 subject to configured recall/F1 guards. When enabled, every validation recording must also have nondecreasing F1 and recall loss <= 0.01.',
        'test_not_evaluated': True})
    write(output / 'run.json', {'seed': SEED, 'epochs': epochs, 'updates_per_epoch': 80,
        'parameters': sum(p.numel() for p in model.parameters()), 'training_clips': len(training),
        'positive_loss_weight': positive_weight,
        'training_candidates': len(all_x), 'validation_clips': len(validation),
        'trainable': 'All verifier weights; Basic Pitch frozen', 'features': list(FEATURE_NAMES),
        'ambiguity': 'Exclude unmatched same-pitch near-onset/overlapping events from training loss only',
        'torch_version': str(torch.__version__)})
    (output / 'split.json').write_bytes((directory / 'note-verifier-split.json').read_bytes())
    print(json.dumps({'selection': chosen, 'epoch': best_epoch, 'validation': selection['aggregate']}), flush=True)


def frozen(directory, run_name='note-verifier-v1'):
    output = directory / run_name
    selection = json.loads((output / 'selection.json').read_text())
    digest = hashlib.sha256((output / 'candidate.pt').read_bytes()).hexdigest()
    if selection['checkpoint_sha256'] != digest:
        raise ValueError('Candidate changed after freeze')
    checkpoint = torch.load(output / 'candidate.pt', map_location='cpu', weights_only=True)
    model = NoteVerifier().eval()
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint, digest


def test(directory, run_name='note-verifier-v1'):
    output = directory / run_name / 'performer-test.json'
    if output.exists():
        raise ValueError('Test consumed; do not tune on this test split')
    model, checkpoint, digest = frozen(directory, run_name)
    items = load_data(directory, 'test')
    result = evaluate(items, probabilities(model, items, checkpoint['mean'].numpy(), checkpoint['scale'].numpy()), checkpoint['threshold'])
    row = result['aggregate']['guitarset']
    result['passes'] = (passes_guard(result) and row['trained']['precision'] > row['baseline']['precision']
                        and row['trained']['false_positives'] <= .9 * row['baseline']['false_positives'])
    result['checkpoint_sha256'] = digest
    write(output, result)
    print(json.dumps({'aggregate': result['aggregate'], 'passes': result['passes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--recall-margin', type=float, default=.01)
    parser.add_argument('--positive-weight', type=float, default=2.)
    parser.add_argument('--protect-recordings', action='store_true')
    parser.add_argument('--run', default='note-verifier-v1')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    (test(args.directory, args.run) if args.test else train(args.directory, args.epochs, args.run,
        args.recall_margin, args.positive_weight, args.protect_recordings))
