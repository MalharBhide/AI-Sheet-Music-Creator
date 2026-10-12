"""Broaden bass KEEP supervision with licensed guitar and original timbres.

The earlier eight bass stress cases are consumed regression, never fitting or
selection data. No reserved performers, Slakh groups or user uploads are inferred.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from bass_training_data import VERSION, cache_one
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from slakh_references import crop

SEED = 261051
RATE = 22050


def original(seed):
    """Random source phrases and timbres, distinct from the fixed regression score."""
    rng = np.random.default_rng(seed)
    notes, cursor = [], 2.75
    root = int(rng.integers(24, 44))
    beat = float(rng.choice((.3, .4, .5, .6, .75)))
    previous = root
    while cursor < 25.5:
        pitch = previous if rng.random() < .35 else root + int(rng.choice((0, 3, 5, 7, 12)))
        spacing = beat * float(rng.choice((.5, 1., 1.5, 2., 4., 8.)))
        end = min(27., cursor + spacing * float(rng.uniform(.55, 1.)))
        notes.append((cursor, end, pitch))
        if pitch + 12 < 60 and rng.random() < .2:
            notes.append((cursor, end, pitch + 12))
        previous = pitch
        cursor += spacing + (beat if rng.random() < .12 else 0.)
    samples = np.zeros(30 * RATE, dtype=np.float32)
    slope = float(rng.uniform(.65, 2.4))
    harmonics = int(rng.integers(2, 13))
    decay = float(rng.choice((.4, 1., 2., 8., 100.)))
    rise = float(rng.uniform(.002, .025))
    for start, end, pitch in notes:
        first, stop = round(start * RATE), round(end * RATE)
        t = np.arange(stop - first) / RATE
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        tone = sum(np.sin(2 * np.pi * frequency * harmonic * t + rng.uniform(-.5, .5)) / harmonic ** slope
                   for harmonic in range(1, harmonics + 1))
        envelope = np.minimum(1., t / rise) * np.minimum(1., np.maximum(0., (end - start - t) / .03))
        envelope *= .25 + .75 * np.exp(-t / decay)
        samples[first:stop] += (tone * envelope * rng.uniform(.06, .2)).astype(np.float32)
    return samples, np.asarray(sorted(notes), dtype=float).reshape(-1, 3)


def prepare(directory):
    base = directory / 'note-verifier-v3-data'
    manifest_path = base / 'note-verifier-split.json'
    prior = json.loads(manifest_path.read_text())
    if prior['guitarset_record'] != '3371780':
        raise ValueError('Changed licensed GuitarSet revision')
    selected = [{**item, 'group': group} for group in ('train', 'validation')
                for item in prior['tracks'][group] if item['corpus'] == 'guitarset']
    if {group: sum(i['group'] == group for i in selected) for group in ('train', 'validation')} != {'train': 239, 'validation': 58}:
        raise ValueError('Changed established GuitarSet performer/exclusion split')
    if any(int(i['player']) not in (range(4) if i['group'] == 'train' else [4]) for i in selected):
        raise ValueError('Reserved performer cannot enter bass fitting')
    output = directory / 'bass-positive-data-v1'
    output.mkdir(exist_ok=True)
    plan = {'seed': SEED, 'original_training_seeds': list(range(SEED, SEED + 64)),
            'original_validation_seeds': list(range(SEED + 64, SEED + 80)),
            'guitar_manifest_sha256': digest(manifest_path), 'guitarset_record': 3371780,
            'guitarset_license': 'CC BY 4.0', 'original_rights': 'Project-authored procedural signals and note sequences',
            'code_sha256': digest(Path(__file__)),
            'cache_sha256': digest(Path(__file__).with_name('bass_training_data.py')),
            'sources': [{'id': i['id'], 'group': i['group'], 'audio_sha256': digest(base / i['audio'])} for i in selected],
            'candidate_decoder': VERSION, 'scope': 'Original timbre/source seeds plus existing licensed GuitarSet train/validation performers; shared compositions and upstream pretraining overlap not ruled out.',
            'no_test_inference': True, 'no_user_audio_or_scores': True,
            'no_consumed_stress_fitting': True}
    preserve(output / 'plan.json', plan)
    items, hashes = [], {}
    jobs = [('original', seed, 'train' if seed < SEED + 64 else 'validation') for seed in range(SEED, SEED + 80)]
    jobs.extend(('guitar', item, item['group']) for item in selected)
    for kind, source, group in jobs:
        name = f'bass-positive-v1-original-{source}' if kind == 'original' else 'bass-positive-v1-' + source['id']
        work = output / name
        work.mkdir(exist_ok=True)
        completion = work / 'complete.json'
        if completion.exists():
            record = json.loads(completion.read_text())
            item = record['item']
            if (record['plan_sha256'] != digest(output / 'plan.json')
                    or record['audio_sha256'] != digest(Path(item['audio']))
                    or record['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                raise ValueError('Changed completed bass-positive evidence')
        else:
            clock = None
            if kind == 'original':
                samples, labels = original(source)
                _, clock = encode(samples, RATE, work, (48, 128, 192)[source % 3])
                audio = work / 'decoded.wav'
                source_group = 'original-bass-seed-' + str(source)
                corpus = 'original-bass-timbres'
            else:
                samples, rate = librosa.load(base / source['audio'], sr=RATE, duration=30)
                if rate != RATE or not RATE <= len(samples) <= RATE * 30 or not np.isfinite(samples).all():
                    raise ValueError('Licensed guitar crop does not match its bounded waveform clock')
                audio = work / 'source.wav'
                sf.write(audio, samples, RATE, subtype='FLOAT')
                labels = np.asarray(source['reference'], dtype=float).reshape(-1, 3)
                labels = labels[(labels[:, 2] >= 21) & (labels[:, 2] < 60)]
                source_group = 'guitar-player-' + source['player']
                corpus = 'guitarset-bass-register'
            duration = sf.info(audio).duration
            reference, support = crop(labels, labels, 0., duration)
            item = {'id': name, 'group': group, 'source_group': source_group, 'corpus': corpus,
                    'audio': str(audio.resolve()), 'reference': reference.tolist(), 'pitch_reference': support.tolist(),
                    'evaluation_window': [.25, duration - .25], 'candidate_decoder': VERSION}
            cache = cache_one(output, item)
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                      'audio_sha256': digest(audio), 'cache_sha256': digest(cache), 'codec_clock': clock}
            preserve(completion, record)
        items.append(item)
        hashes[name] = record['cache_sha256']
        print(json.dumps({'cached': len(items), 'total': len(jobs), 'id': name, 'group': group}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': hashes,
             'plan_sha256': digest(output / 'plan.json'), 'scope': plan['scope'], 'no_test_inference': True,
             'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
