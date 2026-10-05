"""Export a single residual winner after preservation and positive fresh gain."""

import argparse
import json
from pathlib import Path

from app.services.bass_residual import ACOUSTIC_NAMES, NAMES, VERSION
from bass_residual_coverage_data import acoustic_view, load_cached
from bass_residual_coverage_release import load_frozen
from bass_residual_fresh_stress import load_items, require_regression
from export_context_verifier import export
from prepare_robust_training_stems import digest, preserve


def require_fresh(run, fresh, winner):
    report = json.loads((run / 'original-first-pass.json').read_text())
    if (not report['passes'] or not report['per_recording'] or any(not r['passes'] for r in report['per_recording'])
            or report['false_notes_removed'] <= 0 or not report['first_pass_complete']
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or report['test_manifest_sha256'] != digest(fresh / 'manifest.json')
            or report['threshold'] != winner['threshold'] or report['guardian_threshold'] != winner['guardian_threshold']):
        raise ValueError('Failed or changed first-pass residual evidence')
    load_items(fresh)
    return report


def run(source, fresh):
    plan, winner, models = load_frozen(source)
    require_regression(source, winner)
    require_fresh(source, fresh, winner)
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    validation = load_cached(Path(plan['baseline_cache_root']), ('validation',))
    paths = (output / 'bass-residual-v1.npz', output / 'bass-residual-guardian-v1.npz')
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
              'sha256': {path.name: digest(path) for path in paths}, 'tolerance': 1e-12,
              'scope': 'Frozen validation sklearn/portable probabilities; arrays only, no production routing.'}
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    run(args.source, args.fresh)
