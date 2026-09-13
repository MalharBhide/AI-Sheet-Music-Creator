"""Train/validate a small melody model, then explicitly run a frozen held-out test."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import torch
from torch.nn import functional as F

from app.services.melody_decoder import FEATURE_VERSION, RATE, MelodyDecoder, decode, labels

SEED = 1729


def make_split(corpus):
    rows = list(csv.DictReader((corpus / 'vocadito_metadata.csv').open()))
    singers = sorted({row['singer_id'] for row in rows}, key=lambda s: int(s[1:]))
    random.Random(SEED).shuffle(singers)
    groups = {'train': singers[:-10], 'validation': singers[-10:-5], 'test': singers[-5:]}
    return {'seed': SEED, 'group': 'singer_id', 'singers': groups,
            'tracks': {split: [int(row['track_id']) for row in rows if row['singer_id'] in group]
                       for split, group in groups.items()}}


def load_data(directory, ids):
    result = []
    for track in ids:
        with np.load(directory / 'features' / f'vocadito_{track}.npz') as data:
            if str(data['version']) != FEATURE_VERSION:
                raise ValueError('Feature cache version mismatch')
            item = {key: data[key] for key in ['x', 'A1', 'A2', 'baseline']}
        item['id'] = track
        item['y'], item['attacks'] = labels(item['A1'], len(item['x']))
        result.append(item)
    return result


def metrics(reference, estimated, length):
    target, _ = labels(reference, length)
    prediction = np.full(length, 88)
    clock = np.arange(length) / RATE
    for start, end, pitch in estimated:
        prediction[(clock >= start) & (clock < end)] = int(pitch) - 21
    voiced = target != 88
    silent = ~voiced
    reference_intervals = np.stack([reference[:, 0], reference[:, 0] + reference[:, 2]], axis=1)
    estimates = estimated[:, :2].reshape(-1, 2)
    estimate_hz = librosa.midi_to_hz(estimated[:, 2])
    scores = {}
    for name, offset in [('notes', None), ('notes_with_offsets', .2)]:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals, reference[:, 1], estimates, estimate_hz,
            onset_tolerance=.05, pitch_tolerance=50, offset_ratio=offset,
            offset_min_tolerance=.05)
        scores[name] = {'precision': precision, 'recall': recall, 'f1': f1}
    scores.update({
        'voiced_pitch_recall': float(np.mean(prediction[voiced] == target[voiced])),
        'false_silence': float(np.mean(prediction[voiced] == 88)),
        'false_voicing': float(np.mean(prediction[silent] != 88)) if silent.any() else 0,
        'reference_notes': len(reference), 'estimated_notes': len(estimated),
    })
    return scores


def aggregate(rows):
    keys = ['voiced_pitch_recall', 'false_silence', 'false_voicing']
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    for name in ['notes', 'notes_with_offsets']:
        result[name] = {key: float(np.mean([row[name][key] for row in rows]))
                        for key in ['precision', 'recall', 'f1']}
    return result


@torch.no_grad()
def predict(model, dataset):
    model.eval()
    outputs = []
    for item in dataset:
        logits, attacks = model(torch.from_numpy(item['x'])[None])
        outputs.append((logits[0].numpy(), torch.sigmoid(attacks[0]).numpy()))
    return outputs


def evaluate(dataset, outputs, config, annotator='A1'):
    rows = []
    for item, (logits, attacks) in zip(dataset, outputs, strict=True):
        candidate = decode(logits, attacks, **config)
        rows.append({'track': item['id'],
                     'baseline': metrics(item[annotator], item['baseline'], len(item['x'])),
                     'trained': metrics(item[annotator], candidate, len(item['x']))})
    return {'aggregate': {name: aggregate([row[name] for row in rows])
                           for name in ['baseline', 'trained']}, 'tracks': rows}


def train(directory, epochs):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run = directory / 'melody-v1'
    run.mkdir(exist_ok=True)
    if (run / 'test.json').exists():
        raise ValueError('This experiment has consumed its test set; use a new independent corpus for further tuning')
    split = make_split(directory / 'vocadito')
    (run / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    training = load_data(directory, split['tracks']['train'])
    validation = load_data(directory, split['tracks']['validation'])
    model = MelodyDecoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = np.random.default_rng(SEED)
    history, best = [], float('inf')
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'tracks': {key: len(ids) for key, ids in split['tracks'].items()}}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(60):
            batch = []
            for _ in range(4):
                item = training[int(generator.integers(len(training)))]
                start = int(generator.integers(max(1, len(item['x']) - 256 + 1)))
                batch.append({key: item[key][start:start + 256] for key in ['x', 'y', 'attacks']})
            x, y, attacks = [torch.from_numpy(np.stack([item[key] for item in batch]))
                              for key in ['x', 'y', 'attacks']]
            logits, onset_logits = model(x)
            pitch_loss = F.cross_entropy(logits.reshape(-1, 89), y.reshape(-1))
            mask = F.one_hot(y.clamp(max=87), 88).float() * (y != 88)[..., None]
            mask = torch.maximum(mask, attacks)
            onset_loss = (F.binary_cross_entropy_with_logits(onset_logits, attacks,
                           pos_weight=torch.tensor(8.), reduction='none') * mask).sum() / mask.sum().clamp(min=1)
            loss = pitch_loss + .3 * onset_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation_loss = np.mean([F.cross_entropy(
                model(torch.from_numpy(item['x'])[None])[0][0], torch.from_numpy(item['y'])).item()
                for item in validation])
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_pitch_loss': float(validation_loss)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({'state_dict': model.state_dict(), 'feature_version': FEATURE_VERSION,
                        'epoch': epoch, 'seed': SEED}, run / 'candidate.pt')
    (run / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    checkpoint = torch.load(run / 'candidate.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    outputs = predict(model, validation)
    choices = []
    for smoothing in [1, 3, 5]:
        for minimum in [.06, .10, .14]:
            for onset_threshold in [.4, .6, .8, 1.0]:
                config = dict(smoothing=smoothing, minimum=minimum, onset_threshold=onset_threshold)
                report = evaluate(validation, outputs, config)
                score = report['aggregate']['trained']['notes']['f1']
                choices.append((score, config, report))
    _, config, report = max(choices, key=lambda item: item[0])
    checkpoint['decoder'] = config
    torch.save(checkpoint, run / 'candidate.pt')
    (run / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (run / 'selection.json').write_text(json.dumps({
        'epoch': checkpoint['epoch'], 'decoder': config,
        'checkpoint_sha256': hashlib.sha256((run / 'candidate.pt').read_bytes()).hexdigest(),
        'selection': 'Minimum validation frame cross entropy, then maximum validation note onset F1',
        'test_evaluated': False,
    }, indent=2) + '\n')
    print(json.dumps({'selected': config, 'validation': report['aggregate']}), flush=True)


def test(directory):
    torch.set_num_threads(2)
    run = directory / 'melody-v1'
    if (run / 'test.json').exists():
        raise ValueError('Held-out test already exists; do not tune against this test set')
    split = json.loads((run / 'split.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    digest = hashlib.sha256((run / 'candidate.pt').read_bytes()).hexdigest()
    if digest != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after validation selection')
    checkpoint = torch.load(run / 'candidate.pt', map_location='cpu', weights_only=True)
    model = MelodyDecoder()
    model.load_state_dict(checkpoint['state_dict'])
    dataset = load_data(directory, split['tracks']['test'])
    outputs = predict(model, dataset)
    report = {'checkpoint_sha256': digest,
              'A1': evaluate(dataset, outputs, checkpoint['decoder'], 'A1'),
              'A2': evaluate(dataset, outputs, checkpoint['decoder'], 'A2')}
    (run / 'test.json').write_text(json.dumps(report, indent=2) + '\n')
    selection['test_evaluated'] = True
    (run / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({key: report[key]['aggregate'] for key in ['A1', 'A2']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if args.test:
        test(args.directory)
    else:
        train(args.directory, args.epochs)
