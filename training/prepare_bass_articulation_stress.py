"""First-pass original timbre seeds for one frozen bass-boundary candidate.

No recordings from the eight consumed bass cases, fitting seeds or user uploads
enter this corpus. New original signals are encoded as MP3 before caching the
actual bounded production detector. No notation or transcription jobs are made.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from bass_articulation_labels import annotate
from bass_articulation_release import load_frozen
from bass_training_data import VERSION, cache_one, load
from current_bass_baseline import prepare as baseline
from prepare_bass_positive_data import RATE, SEED, original
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from train_bass_boundaries import predictions, score

SEEDS = tuple(range(261301, 261333))


def require_regression(run, winner):
    report = json.loads((run / 'consumed-regression.json').read_text())
    if (not report['passes'] or not report['per_recording'] or any(not row['passes'] for row in report['per_recording'])
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or report['threshold'] != winner['threshold'] or report['guardian_threshold'] != winner['guardian_threshold']):
        raise ValueError('Failed or changed consumed boundary regression')
    return digest(run / 'consumed-regression.json')


def prepare(run, output):
    frozen, winner, _ = load_frozen(run)
    regression_sha = require_regression(run, winner)
    if set(SEEDS) & set(range(SEED, SEED + 80)):
        raise ValueError('Original stress seeds overlap fitting/validation')
    fitted = {item['source_group'] for root in frozen['manifests']
              for item in json.loads((Path(root) / 'manifest.json').read_text())['items']}
    if any('original-bass-seed-' + str(seed) in fitted for seed in SEEDS):
        raise ValueError('Original stress source already fitted')
    output.mkdir(exist_ok=True)
    plan = {'seeds': list(SEEDS), 'rights': 'Project-authored procedural notes and signals',
            'rate': RATE, 'seconds': 30., 'candidate_decoder': VERSION,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in (
                'prepare_bass_articulation_stress.py', 'prepare_bass_positive_data.py',
                'prepare_slakh_bass.py', 'bass_training_data.py')},
            'checkpoint_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'regression_sha256': regression_sha,
            'scope': 'New source/timbre seeds from original generator, same distribution as fitting. First-pass stress only, not independent human music or commercial-song accuracy.',
            'no_fitting_selection_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    items, cache_hashes = [], {}
    for seed in SEEDS:
        name = 'bass-articulation-stress-v2-' + str(seed)
        work = output / name
        work.mkdir(exist_ok=True)
        completion = work / 'complete.json'
        if completion.exists():
            record = json.loads(completion.read_text())
            item = record['item']
            if (record['plan_sha256'] != digest(output / 'plan.json')
                    or record['audio_sha256'] != digest(Path(item['audio']))
                    or record['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                raise ValueError('Changed completed original boundary stress')
        else:
            samples, labels = original(seed)
            bitrate = (48, 128, 192)[seed % 3]
            _, clock = encode(samples, RATE, work, bitrate)
            audio = work / 'decoded.wav'
            if sf.info(audio).duration != 30.:
                raise ValueError('Changed original stress audio clock')
            item = {'id': name, 'group': 'test', 'source_group': 'original-bass-seed-' + str(seed),
                    'corpus': 'original-bass-articulation-stress', 'audio': str(audio.resolve()),
                    'reference': labels.tolist(), 'pitch_reference': labels.tolist(),
                    'evaluation_window': [.25, 29.75], 'candidate_decoder': VERSION}
            cache = cache_one(output, item)
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                      'audio_sha256': digest(audio), 'cache_sha256': digest(cache),
                      'codec_clock': clock, 'bitrate_kbps': bitrate}
            preserve(completion, record)
        items.append(item)
        cache_hashes[name] = record['cache_sha256']
        print(json.dumps({'cached': len(items), 'total': len(SEEDS), 'seed': seed}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': cache_hashes,
             'plan_sha256': digest(output / 'plan.json'), 'scope': plan['scope'],
             'no_fitting_selection_user_audio_or_scores': True})


def evaluate(run, data):
    destination = run / 'original-first-pass.json'
    if destination.exists():
        raise ValueError('Preserve consumed original boundary evaluation; no retuning')
    _, winner, models = load_frozen(run)
    regression_sha = require_regression(run, winner)
    plan = json.loads((data / 'plan.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    if (plan['seeds'] != list(SEEDS) or manifest['plan_sha256'] != digest(data / 'plan.json')
            or plan['checkpoint_sha256'] != winner['checkpoint_sha256']
            or plan['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or plan['regression_sha256'] != regression_sha
            or [item['source_group'] for item in manifest['items']] != ['original-bass-seed-' + str(seed) for seed in SEEDS]
            or any(item['group'] != 'test' for item in manifest['items'])):
        raise ValueError('Changed frozen original boundary stress plan')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Changed original boundary preparation code')
    items = annotate(baseline(load(data, 'test')))
    result = score(items, predictions(models, items), winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  test_manifest_sha256=digest(data / 'manifest.json'),
                  scope=plan['scope'], no_retuning=True,
                  first_pass_complete=True, now_consumed_regression=True)
    preserve(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    (evaluate if args.evaluate else prepare)(args.run, args.output)
