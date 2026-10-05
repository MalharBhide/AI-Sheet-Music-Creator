"""Export only the frozen texture winner with positive, preserved first-pass gain."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_texture import ACOUSTIC_NAMES, NAMES, VERSION
from bass_texture_v12_data import acoustic_view, load_cached
from bass_texture_v12_release import load_frozen
from export_context_verifier import export
from fresh_bass_texture_v12 import SEEDS, require_regression
from prepare_robust_training_stems import digest, preserve


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
            or manifest['plan_sha256'] != digest(fresh / 'plan.json') or plan['seeds'] != list(SEEDS)
            or not plan['positive_first_pass_gain_required'] or not manifest['now_consumed_regression']):
        raise ValueError('Failed or changed first-pass texture evidence')
    root = Path(__file__).resolve().parents[1]
    if any(digest(root / name) != expected for name, expected in plan['code_sha256'].items()):
        raise ValueError('Changed first-pass texture preparation')
    if [r['source_group'] for r in manifest['items']] != ['original-bass-seed-' + str(seed) for seed in SEEDS]:
        raise ValueError('Changed first-pass texture source groups')
    items = []
    for record in manifest['items']:
        path = fresh / record['id'] / 'features.npz'
        if record['group'] != 'test' or digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
            raise ValueError('Changed first-pass texture source/cache')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != manifest['plan_sha256']:
                raise ValueError('Changed first-pass texture cache identity')
            items.append({**record, **{key: saved[key] for key in ('base_x', 'events', 'x', 'eligible')}})
    return items


def run(source, fresh):
    plan, winner, models = load_frozen(source)
    require_regression(source, winner)
    fresh_items(source, fresh, winner)
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    validation = load_cached(Path(plan['baseline_cache_root']), ('validation',))
    paths = (output / 'bass-texture-v1.npz', output / 'bass-texture-guardian-v1.npz')
    for model, threshold, path, names, view in (
        (models[0], winner['threshold'], paths[0], NAMES, lambda x: x),
        (models[1], winner['guardian_threshold'], paths[1], ACOUSTIC_NAMES, acoustic_view),
    ):
        export(model, threshold, path, [{'x': view(item['x'])} for item in validation if len(item['x'])],
               feature_names=names, feature_version=VERSION)
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(validation), 'retained_events': sum(len(i['events']) for i in validation),
              'sha256': {p.name: digest(p) for p in paths}, 'tolerance': 1e-12,
              'scope': 'Frozen validation sklearn/portable probability parity; no production routing.'}
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    run(args.run, args.fresh)
