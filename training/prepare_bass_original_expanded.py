"""Additional original bass timbres, never consumed regressions or user audio."""

import argparse
import json
from pathlib import Path

from bass_training_data import VERSION, cache_one
from prepare_bass_positive_data import RATE, original
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode

SEEDS = tuple(range(261801, 261881))
TRAIN_COUNT = 64


def prepare(directory):
    output = directory / 'bass-original-expanded-v1'
    output.mkdir(exist_ok=True)
    plan = {'seeds': list(SEEDS), 'train_count': TRAIN_COUNT,
            'rights': 'Project-authored procedural signals and note sequences',
            'source_distribution': 'Same original generator as prior fitting, new source/timbre seeds; not independent human-song evidence.',
            'rate': RATE, 'seconds': 30., 'mp3_bitrate_kbps_cycle': [48, 128, 192],
            'candidate_decoder': VERSION, 'no_test_inference': True, 'no_user_audio_or_scores': True,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in (
                'prepare_bass_original_expanded.py', 'prepare_bass_positive_data.py',
                'prepare_slakh_bass.py', 'bass_training_data.py')}}
    preserve(output / 'plan.json', plan)
    items, hashes = [], {}
    for index, seed in enumerate(SEEDS):
        name = 'bass-original-expanded-v1-' + str(seed)
        work = output / name
        work.mkdir(exist_ok=True)
        completion = work / 'complete.json'
        if completion.exists():
            record = json.loads(completion.read_text())
            item = record['item']
            if (record['plan_sha256'] != digest(output / 'plan.json')
                    or record['audio_sha256'] != digest(Path(item['audio']))
                    or record['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                raise ValueError('Changed expanded original source evidence')
        else:
            samples, labels = original(seed)
            _, clock = encode(samples, RATE, work, plan['mp3_bitrate_kbps_cycle'][index % 3])
            audio = work / 'decoded.wav'
            item = {'id': name, 'group': 'train' if index < TRAIN_COUNT else 'validation',
                    'source_group': 'original-bass-seed-' + str(seed), 'corpus': 'original-bass-timbres',
                    'audio': str(audio.resolve()), 'reference': labels.tolist(), 'pitch_reference': labels.tolist(),
                    'evaluation_window': [.25, 29.75], 'candidate_decoder': VERSION}
            path = cache_one(output, item)
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                      'audio_sha256': digest(audio), 'cache_sha256': digest(path), 'codec_clock': clock}
            preserve(completion, record)
        items.append(item)
        hashes[name] = record['cache_sha256']
        print(json.dumps({'cached': len(items), 'total': len(SEEDS), 'id': name, 'group': item['group']}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': hashes,
             'plan_sha256': digest(output / 'plan.json'), 'no_test_inference': True,
             'no_user_audio_or_scores': True, 'scope': plan['source_distribution']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
