"""Export the single frozen temporal CNN after positive preservation gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.temporal_note_model import VERSION, probability
from bass_temporal_data import load_cached
from bass_temporal_release import load_frozen
from fresh_bass_temporal import require_regression
from prepare_robust_training_stems import digest, preserve
from train_bass_temporal import FRESH_SEEDS


def fresh_items(run, fresh, winner):
    result = json.loads((run / 'original-first-pass.json').read_text())
    manifest = json.loads((fresh / 'manifest.json').read_text())
    plan = json.loads((fresh / 'plan.json').read_text())
    if (not result['passes'] or result['false_notes_removed'] <= 0
            or len(result['per_recording']) != 32 or any(not r['passes'] for r in result['per_recording'])
            or not result['first_pass_complete'] or not result['now_consumed_regression']
            or result['checkpoint_sha256'] != winner['checkpoint_sha256']
            or result['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or result['test_manifest_sha256'] != digest(fresh / 'manifest.json')
            or (result['threshold'], result['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])
            or manifest['plan_sha256'] != digest(fresh / 'plan.json') or plan['seeds'] != list(FRESH_SEEDS)
            or not plan['positive_first_pass_gain_required'] or not manifest['now_consumed_regression']):
        raise ValueError('Failed or changed first-pass temporal evidence')
    root = Path(__file__).resolve().parents[1]
    if any(digest(root / name) != expected for name, expected in plan['code_sha256'].items()):
        raise ValueError('Changed first-pass temporal preparation')
    if [r['source_group'] for r in manifest['items']] != ['original-bass-seed-' + str(seed) for seed in FRESH_SEEDS]:
        raise ValueError('Changed first-pass temporal source groups')
    items = []
    for record in manifest['items']:
        path = fresh / record['id'] / 'features.npz'
        if record['group'] != 'test' or digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
            raise ValueError('Changed first-pass temporal source/cache')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != manifest['plan_sha256']:
                raise ValueError('Changed first-pass temporal cache identity')
            items.append({**record, **{key: saved[key] for key in ('base_x', 'events', 'x', 'eligible', 'frames')}})
    return items


def run(source, fresh):
    plan, winner, model, normalizer = load_frozen(source)
    require_regression(source, winner)
    fresh_items(source, fresh, winner)
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    checkpoint = __import__('torch').load(source / winner['checkpoint'], map_location='cpu', weights_only=True)
    arrays = {'version': VERSION, 'format_version': 1, 'width': checkpoint['width'],
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'mean': normalizer[0], 'scale': normalizer[1]}
    arrays.update({'weight_' + name: tensor.numpy() for name, tensor in model.state_dict().items()})
    path = output / 'bass-temporal-v1.npz'
    np.savez_compressed(path, **arrays)
    from app.services.bass_temporal import TemporalCandidateModel

    with np.load(path, allow_pickle=False) as saved:
        portable = TemporalCandidateModel(saved)
    validation = load_cached(Path(plan['sequence_root']), ('validation',))
    error = 0.
    for item in validation:
        p = probability(model, item['frames'], item['x'], normalizer)
        q = portable.probability(item['frames'], item['x'])
        np.testing.assert_array_equal(p, q, err_msg=item['id'])
        error = max(error, float(np.max(np.abs(p-q), initial=0.)))
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(validation), 'v14_retained_events': sum(len(i['events']) for i in validation),
              'sha256': {path.name: digest(path)}, 'maximum_probability_error': error,
              'scope': 'Frozen validation Torch/NPZ probability parity; no production routing.'}
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    run(args.run, args.fresh)
