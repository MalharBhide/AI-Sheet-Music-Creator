"""Train on commercial-compatible corpora; select on validation, test once."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from melody_candidate import ContextMelodyDecoder
from torch.nn import functional as F
from train_melody import evaluate, load_data, predict

from app.services.melody_decoder import FEATURE_VERSION, MelodyDecoder, decode, labels

SEED = 260914
ASSET = Path(__file__).resolve().parents[1] / 'backend/app/assets/vocal-melody-v1.pt'


def load_vocalset(directory, group):
    split = json.loads((directory / 'vocalset-split.json').read_text())
    if split.get('annotation_clock_version') != 1:
        raise ValueError('Reconcile dataset annotation clocks before training or evaluation')
    items = []
    for entry in split['tracks'][group]:
        with np.load(directory / 'vocalset-features' / (entry['id'] + '.npz')) as cache:
            if str(cache['version']) != FEATURE_VERSION:
                raise ValueError('Feature version mismatch')
            item = {'id': entry['id'], 'x': cache['x'], 'A1': cache['reference'],
                    'baseline': cache['baseline']}
        item['y'], item['attacks'] = labels(item['A1'], len(item['x']))
        items.append(item)
    return items


def baseline_model():
    checkpoint = torch.load(ASSET, map_location='cpu', weights_only=True)
    model = MelodyDecoder().eval()
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint


def vocadito(directory, group):
    split = json.loads((directory / 'melody-v1/split.json').read_text())
    items = load_data(directory, split['tracks'][group])
    model, checkpoint = baseline_model()
    for item, (logits, attacks) in zip(items, predict(model, items), strict=True):
        item['baseline'] = decode(logits, attacks, **checkpoint['decoder'])
    return items


def batch(corpora, rng, size=4, length=256):
    # Equal probability per corpus prevents long scale recordings from
    # overwhelming the smaller collection of natural song phrases.
    x = np.zeros((size, length, 88, 10), dtype=np.float32)
    y = np.full((size, length), -100, dtype=np.int64)
    attacks = np.zeros((size, length, 88), dtype=np.float32)
    for index in range(size):
        corpus = corpora[int(rng.integers(len(corpora)))]
        item = corpus[int(rng.integers(len(corpus)))]
        start = int(rng.integers(max(1, len(item['x']) - length + 1)))
        count = min(length, len(item['x']) - start)
        x[index, :count] = item['x'][start:start + count]
        y[index, :count] = item['y'][start:start + count]
        attacks[index, :count] = item['attacks'][start:start + count]
    return tuple(torch.from_numpy(value) for value in (x, y, attacks))


def loss(model, x, y, attacks):
    logits, onset_logits = model(x)
    pitch_loss = F.cross_entropy(logits.reshape(-1, 89), y.reshape(-1), ignore_index=-100)
    valid = y != -100
    mask = F.one_hot(y.clamp(min=0, max=87), 88).float() * ((y != 88) & valid)[..., None]
    mask = torch.maximum(mask, attacks) * valid[..., None]
    onset_loss = (F.binary_cross_entropy_with_logits(onset_logits, attacks,
                  pos_weight=torch.tensor(8.), reduction='none') * mask).sum() / mask.sum().clamp(min=1)
    return pitch_loss + .3 * onset_loss


def train(directory, epochs, *, rehearsal=False):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    run = directory / 'context-melody-v2'
    run.mkdir(exist_ok=True)
    if (run / 'selection.json').exists() or (run / 'test.json').exists():
        raise ValueError('Frozen experiment already exists; do not overwrite or tune against its test')
    training = [vocadito(directory, 'train'), load_vocalset(directory, 'train')]
    validation = [vocadito(directory, 'validation'), load_vocalset(directory, 'validation')]
    teacher, checkpoint = baseline_model()
    model = ContextMelodyDecoder()
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert not unexpected and all(name.startswith('context.') for name in missing)
    if rehearsal:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('context.'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.0003, weight_decay=.0001)
    history, best = [], float('inf')
    metadata = {'seed': SEED, 'parameters': sum(p.numel() for p in model.parameters()),
                'architecture': 'V1 plus pitch-shared 1.26 second temporal residual branch',
                'warm_start_sha256': hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                'train_tracks': [len(c) for c in training], 'validation_tracks': [len(c) for c in validation],
                'train_seconds': [sum(len(i['x']) / 50 for i in c) for c in training],
                'epochs': epochs, 'updates_per_epoch': 60, 'batch': 4, 'window_frames': 256,
                'optimizer': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001,
                'sampling': 'Equal corpus probability, then uniform recordings and crops',
                'augmentation': 'None; natural variation from distinct singers and techniques',
                'rehearsal': rehearsal,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'rehearsal_loss': 'Frozen V1 backbone, 2x frame KL plus 0.2x voiced attack-logit MSE on an additional Vocadito training crop per update' if rehearsal else None,
                'test_gate': 'VocalSet note and offset F1 improve; pitch recall does not regress; false silence increases by at most 0.02. Vocadito validation note F1 regression at most 0.02.'}
    (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(60):
            value = loss(model, *batch(training, rng))
            if rehearsal:
                # Rehearse only TRAINING phrases. This constrains catastrophic
                # forgetting without using validation or test labels to fit.
                old_x, old_y, _ = batch([training[0]], rng, size=1)
                with torch.no_grad():
                    old_logits, old_attacks = teacher(old_x)
                new_logits, new_attacks = model(old_x)
                valid = old_y != -100
                divergence = F.kl_div(F.log_softmax(new_logits, dim=-1),
                                      F.softmax(old_logits, dim=-1), reduction='none').sum(dim=-1)
                pitch_mask = F.one_hot(old_y.clamp(min=0, max=87), 88) * ((old_y != 88) & valid)[..., None]
                attack_difference = ((new_attacks - old_attacks).square() * pitch_mask).sum() / pitch_mask.sum().clamp(min=1)
                value = value + 2 * divergence[valid].mean() + .2 * attack_difference
            optimizer.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            losses.append(value.item())
        model.eval()
        with torch.inference_mode():
            validation_losses = [float(np.mean([F.cross_entropy(
                model(torch.from_numpy(item['x'])[None])[0][0], torch.from_numpy(item['y'])).item()
                for item in corpus])) for corpus in validation]
        score = float(np.mean(validation_losses))
        record = {'epoch': epoch, 'training_loss': float(np.mean(losses)),
                  'validation_loss_by_corpus': validation_losses, 'selection_loss': score}
        history.append(record)
        (run / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        if score < best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'feature_version': FEATURE_VERSION,
                        'architecture': 'context-v2', 'epoch': epoch, 'seed': SEED}, run / 'candidate.pt')
    checkpoint = torch.load(run / 'candidate.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    outputs = [predict(model, corpus) for corpus in validation]
    choices = []
    for smoothing in (1, 3, 5):
        for minimum in (.06, .10):
            for threshold in (.8, 1.):
                config = dict(smoothing=smoothing, minimum=minimum, onset_threshold=threshold)
                reports = [evaluate(corpus, output, config) for corpus, output in zip(validation, outputs, strict=True)]
                score = float(np.mean([.5 * (r['aggregate']['trained']['notes']['f1']
                                      + r['aggregate']['trained']['notes_with_offsets']['f1']) for r in reports]))
                choices.append((score, config, reports))
    # Exercise data must not erase the natural-song phrasing learned in V1.
    # Apply the predeclared old-corpus gate during validation selection too.
    eligible = [choice for choice in choices if (
        choice[2][0]['aggregate']['trained']['notes']['f1'] >=
        choice[2][0]['aggregate']['baseline']['notes']['f1'] - .02)]
    _, config, reports = max(eligible or choices, key=lambda choice: choice[0])
    checkpoint['decoder'] = config
    torch.save(checkpoint, run / 'candidate.pt')
    (run / 'validation.json').write_text(json.dumps(dict(zip(['vocadito', 'vocalset'], reports, strict=True)), indent=2) + '\n')
    (run / 'selection.json').write_text(json.dumps({
        'epoch': checkpoint['epoch'], 'decoder': config,
        'checkpoint_sha256': hashlib.sha256((run / 'candidate.pt').read_bytes()).hexdigest(),
        'selection': 'Lowest equally weighted corpus validation pitch CE; then highest mean onset/offset F1 across both validation corpora, preferring configs that pass the Vocadito validation regression gate',
        'test_evaluated': False}, indent=2) + '\n')
    print(json.dumps({'selected': config, 'validation': [r['aggregate'] for r in reports]}), flush=True)


def test(directory):
    torch.set_num_threads(2)
    run = directory / 'context-melody-v2'
    if (run / 'test.json').exists():
        raise ValueError('This held-out test has already been consumed')
    selection = json.loads((run / 'selection.json').read_text())
    digest = hashlib.sha256((run / 'candidate.pt').read_bytes()).hexdigest()
    if digest != selection['checkpoint_sha256']:
        raise ValueError('Candidate changed after selection')
    checkpoint = torch.load(run / 'candidate.pt', map_location='cpu', weights_only=True)
    model = ContextMelodyDecoder()
    model.load_state_dict(checkpoint['state_dict'])
    dataset = load_vocalset(directory, 'test')
    report = evaluate(dataset, predict(model, dataset), checkpoint['decoder'])
    report['checkpoint_sha256'] = digest
    scores = report['aggregate']
    candidate, baseline = scores['trained'], scores['baseline']
    old_validation = json.loads((run / 'validation.json').read_text())['vocadito']['aggregate']
    gates = {
        'note_f1': candidate['notes']['f1'] > baseline['notes']['f1'],
        'offset_f1': candidate['notes_with_offsets']['f1'] > baseline['notes_with_offsets']['f1'],
        'pitch_recall': candidate['voiced_pitch_recall'] >= baseline['voiced_pitch_recall'],
        'false_silence': candidate['false_silence'] <= baseline['false_silence'] + .02,
        'false_voicing': candidate['false_voicing'] <= baseline['false_voicing'] + .02,
        'vocadito_validation': old_validation['trained']['notes']['f1'] >= old_validation['baseline']['notes']['f1'] - .02,
        'vocadito_offsets': old_validation['trained']['notes_with_offsets']['f1'] >= old_validation['baseline']['notes_with_offsets']['f1'] - .02,
        'vocadito_false_voicing': old_validation['trained']['false_voicing'] <= old_validation['baseline']['false_voicing'] + .02,
    }
    report['promotion_gates'] = gates
    report['eligible_for_promotion'] = all(gates.values())
    (run / 'test.json').write_text(json.dumps(report, indent=2) + '\n')
    selection['test_evaluated'] = True
    (run / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({'test': scores, 'gates': gates}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--rehearsal', action='store_true')
    args = parser.parse_args()
    if args.test:
        test(args.directory)
    else:
        train(args.directory, args.epochs, rehearsal=args.rehearsal)
