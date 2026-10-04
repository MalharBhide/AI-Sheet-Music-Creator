"""Original stress cases, only inferred after freezing a positive bass winner."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from bass_release import load_frozen
from bass_training_data import VERSION, cache_one
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode

RATE = 22050
SECONDS = 30
CASES = ('held', 'repeated', 'triplet', 'octave', 'low', 'legato', 'sparse', 'syncopated')


def score(case):
    if case == 'held':
        return [(3., 11., 36), (12., 19., 43), (20., 27., 40)]
    if case == 'repeated':
        return [(float(start), float(start + .36), 40) for start in np.arange(3., 27., .5)]
    if case == 'triplet':
        return [(3. + index / 3, 3. + index / 3 + .22, (36, 43, 48)[index % 3]) for index in range(66)]
    if case == 'octave':
        return [(float(start), float(start + 1.8), pitch) for start in np.arange(3., 27., 2.5) for pitch in (36, 48)]
    if case == 'low':
        return [(3. + index * 1.5, 4. + index * 1.5, (24, 28, 31, 33)[index % 4]) for index in range(16)]
    if case == 'legato':
        return [(3. + index, 4. + index, (40, 43, 45, 47)[index % 4]) for index in range(24)]
    if case == 'sparse':
        return [(3., 5., 36), (12., 16., 40), (24., 27., 43)]
    if case == 'syncopated':
        return [(3. + index * .75, 3. + index * .75 + .4, (40, 45, 47)[index % 3]) for index in range(32)]
    raise ValueError('Unknown predeclared bass stress case')


def render(notes, case_index):
    waveform = np.zeros(RATE * SECONDS, dtype=np.float32)
    for start, end, pitch in notes:
        first, stop = round(start * RATE), round(end * RATE)
        t = np.arange(stop - first) / RATE
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        tone = np.zeros(len(t))
        for harmonic in range(1, 9):
            tone += (np.sin(2 * np.pi * frequency * harmonic * t + .17 * harmonic)
                     / harmonic ** (1.2 + case_index * .1))
        envelope = np.minimum(1., t / .008) * np.minimum(1., np.maximum(0., (end - start - t) / .025))
        waveform[first:stop] += (tone * envelope * .16).astype(np.float32)
    return waveform


def prepare(directory, run):
    winner, _ = load_frozen(directory, run)
    output = directory.parent / 'bass-held-regression-v1'
    output.mkdir(exist_ok=True)
    plan = {'cases': list(CASES), 'rate': RATE, 'seconds': SECONDS,
            'code_sha256': digest(Path(__file__)), 'codec_sha256': digest(Path(__file__).with_name('prepare_slakh_bass.py')),
            'candidate_decoder': VERSION, 'checkpoint_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'scope': 'Original procedural stress; distinct source groups; never fitted or used to choose threshold/checkpoint.',
            'no_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    items, hashes = [], {}
    for index, case in enumerate(CASES):
        name = 'bass-held-v1-' + case
        work = output / name
        work.mkdir(exist_ok=True)
        completion = work / 'complete.json'
        if completion.exists():
            record = json.loads(completion.read_text())
            item = record['item']
            if (record['plan_sha256'] != digest(output / 'plan.json')
                    or record['audio_sha256'] != digest(Path(item['audio']))
                    or record['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                raise ValueError('Changed completed bass stress evidence')
        else:
            notes = score(case)
            waveform = render(notes, index)
            bitrate = (48, 128, 192)[index % 3]
            _, clock = encode(waveform, RATE, work, bitrate)
            duration = sf.info(work / 'decoded.wav').duration
            if duration != SECONDS:
                raise ValueError('Changed stress audio clock')
            item = {'id': name, 'group': 'test', 'source_group': name, 'corpus': 'original-bass-stress',
                    'audio': str((work / 'decoded.wav').resolve()), 'reference': notes, 'pitch_reference': notes,
                    'evaluation_window': [.25, SECONDS - .25], 'candidate_decoder': VERSION}
            # Normalize tuples before cache identity and completion JSON roundtrip.
            item = json.loads(json.dumps(item))
            cache = cache_one(output, item)
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                      'audio_sha256': digest(Path(item['audio'])), 'cache_sha256': digest(cache),
                      'codec_clock': clock, 'bitrate_kbps': bitrate}
            preserve(completion, record)
        items.append(item)
        hashes[name] = record['cache_sha256']
        print(json.dumps({'cached': index + 1, 'total': len(CASES), 'case': case}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': hashes,
             'plan_sha256': digest(output / 'plan.json'), 'scope': plan['scope'], 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.run)
