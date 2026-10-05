"""First-pass unseen original MP3 seeds for one frozen V11 residual winner."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_verifier import BassVerifier
from bass_residual_coverage_release import load_frozen
from bass_residual_v11_data import features
from bass_training_data import VERSION, cache_one, eligible
from current_bass_v11_baseline import decisions
from prepare_bass_positive_data import RATE, original
from prepare_bass_v11_baseline import native_evidence
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from train_bass_residual_coverage import predictions, score

SEEDS = tuple(range(261901, 261933))


def require_regression(run, winner):
    report = json.loads((run / 'consumed-regression.json').read_text())
    if (not report['passes'] or not report['per_recording'] or any(not r['passes'] for r in report['per_recording'])
            or report['false_notes_removed'] <= 0
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or report['threshold'] != winner['threshold'] or report['guardian_threshold'] != winner['guardian_threshold']):
        raise ValueError('Failed or changed consumed residual regression')
    return digest(run / 'consumed-regression.json')


def prepare(run, output):
    frozen, winner, _ = load_frozen(run)
    regression = require_regression(run, winner)
    used = {i['source_group'] for root in [*frozen['manifests'], *frozen['consumed_regression_manifests']]
            for i in json.loads((Path(root) / 'manifest.json').read_text())['items']}
    if any('original-bass-seed-' + str(seed) in used for seed in SEEDS):
        raise ValueError('Fresh residual seeds overlap fitting or consumed regression')
    output.mkdir(exist_ok=True)
    post = output / 'post-v11-features'
    post.mkdir(exist_ok=True)
    plan = {'seeds': list(SEEDS), 'rights': 'Project-authored procedural signals and note sequences',
            'scope': 'Unseen source/timbre seeds from existing generator; first-pass MP3 stress, not independent human songs.',
            'candidate_decoder': VERSION, 'checkpoint_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'regression_sha256': regression, 'positive_first_pass_gain_required': True,
            'no_fitting_selection_user_audio_or_scores': True,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in (
                'bass_residual_fresh_stress.py', 'prepare_bass_positive_data.py',
                'prepare_slakh_bass.py', 'bass_training_data.py', 'prepare_bass_v11_baseline.py',
                'current_bass_v11_baseline.py', 'bass_residual_v11_data.py')}}
    preserve(output / 'plan.json', plan)
    items, raw_hashes, post_hashes = [], {}, {}
    bass = BassVerifier()
    for index, seed in enumerate(SEEDS):
        name = 'bass-residual-fresh-v1-' + str(seed)
        work = output / name
        work.mkdir(exist_ok=True)
        complete = work / 'complete.json'
        if complete.exists():
            record = json.loads(complete.read_text())
            item = record['item']
            if (record['plan_sha256'] != digest(output / 'plan.json')
                    or record['audio_sha256'] != digest(Path(item['audio']))
                    or record['raw_cache_sha256'] != digest(output / 'features' / (name + '.npz'))
                    or record['post_cache_sha256'] != digest(post / (name + '.npz'))):
                raise ValueError('Changed completed first-pass residual evidence')
        else:
            samples, labels = original(seed)
            _, clock = encode(samples, RATE, work, (48, 128, 192)[index % 3])
            audio = work / 'decoded.wav'
            item = {'id': name, 'group': 'test', 'source_group': 'original-bass-seed-' + str(seed),
                    'corpus': 'original-bass-residual-fresh', 'audio': str(audio.resolve()),
                    'reference': labels.tolist(), 'pitch_reference': labels.tolist(),
                    'evaluation_window': [.25, 29.75], 'candidate_decoder': VERSION}
            path = cache_one(output, item)
            with np.load(path, allow_pickle=False) as saved:
                raw = {**item, **{field: saved[field] for field in ('x', 'events', 'eligible')}}
                duration = float(saved['duration'])
            result = decisions(raw, duration)
            x = native_evidence(raw, result) if result['merged_boundaries'] else raw['x'][result['raw_indices']]
            p, g = [model.probability(x[:, :model.feature_count]) for model in bass.models]
            np.savez_compressed(post / (name + '.npz'), x=features(result['events'], x, p, g), base_x=x,
                                events=result['events'], eligible=eligible(result['events'], duration),
                                context=p, guardian=g, plan_sha256=digest(output / 'plan.json'))
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'), 'codec_clock': clock,
                      'audio_sha256': digest(audio), 'raw_cache_sha256': digest(path),
                      'post_cache_sha256': digest(post / (name + '.npz')),
                      'v11_merged_boundaries': result['merged_boundaries']}
            preserve(complete, record)
        items.append(item)
        raw_hashes[name], post_hashes[name] = record['raw_cache_sha256'], record['post_cache_sha256']
        print(json.dumps({'cached': len(items), 'total': len(SEEDS), 'id': name}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': raw_hashes,
             'post_cache_sha256': post_hashes, 'plan_sha256': digest(output / 'plan.json'),
             'no_fitting_selection_user_audio_or_scores': True})


def load_items(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    plan = json.loads((data / 'plan.json').read_text())
    if manifest['plan_sha256'] != digest(data / 'plan.json') or plan['seeds'] != list(SEEDS):
        raise ValueError('Changed first-pass residual manifest')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Changed first-pass residual preparation')
    if [i['source_group'] for i in manifest['items']] != ['original-bass-seed-' + str(seed) for seed in SEEDS]:
        raise ValueError('Changed first-pass residual source groups')
    items = []
    for item in manifest['items']:
        path = data / 'post-v11-features' / (item['id'] + '.npz')
        if item['group'] != 'test' or digest(path) != manifest['post_cache_sha256'][item['id']]:
            raise ValueError('Changed first-pass residual feature cache')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != manifest['plan_sha256']:
                raise ValueError('Changed first-pass residual feature identity')
            result = {**item, **{field: saved[field] for field in ('x', 'base_x', 'events', 'eligible')}}
        result.update(reference=np.asarray(item['reference']).reshape(-1, 3),
                      pitch_reference=np.asarray(item['pitch_reference']).reshape(-1, 3), seconds=29.5)
        items.append(result)
    return items


def evaluate(run, data):
    destination = run / 'original-first-pass.json'
    if destination.exists():
        raise ValueError('Preserve first-pass residual evaluation; no retuning')
    _, winner, models = load_frozen(run)
    plan = json.loads((data / 'plan.json').read_text())
    if (plan['checkpoint_sha256'] != winner['checkpoint_sha256']
            or plan['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or plan['regression_sha256'] != require_regression(run, winner)):
        raise ValueError('Changed frozen first-pass residual winner')
    items = load_items(data)
    result = score(items, *predictions(models, items), winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  test_manifest_sha256=digest(data / 'manifest.json'), first_pass_complete=True,
                  now_consumed_regression=True, no_retuning=True, scope=plan['scope'])
    preserve(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    (evaluate if args.evaluate else prepare)(args.run, args.data)
