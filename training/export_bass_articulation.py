"""Export one gated key-release-aware bass winner; no test-driven retuning."""

import argparse
import json
from pathlib import Path

from app.services.bass_articulation import ACOUSTIC_NAMES, RELATION_NAMES, VERSION
from bass_articulation_labels import annotate
from bass_articulation_release import load_frozen
from bass_boundary_evidence import features
from export_context_verifier import export
from prepare_robust_training_stems import digest, preserve
from train_bass_boundaries import collect


def require_report(run, winner, stage, positive=False):
    report = json.loads((run / (stage + '.json')).read_text())
    if (not report['passes'] or not report['per_recording'] or any(not row['passes'] for row in report['per_recording'])
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or report['threshold'] != winner['threshold'] or report['guardian_threshold'] != winner['guardian_threshold']
            or (positive and report['false_notes_removed'] <= 0)):
        raise ValueError('Failed or changed bass articulation evidence')
    return report


def run(source, fresh):
    plan, winner, models = load_frozen(source)
    require_report(source, winner, 'consumed-regression', positive=True)
    first = require_report(source, winner, 'original-first-pass', positive=True)
    if first['test_manifest_sha256'] != digest(fresh / 'manifest.json'):
        raise ValueError('Changed first-pass articulation corpus')
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    _, validation = collect([Path(root) for root in plan['manifests']])
    items = annotate(validation)
    paths = (output / 'bass-articulation-v1.npz', output / 'bass-articulation-guardian-v1.npz')
    for model, threshold, path, names, acoustic in (
        (models[0], winner['threshold'], paths[0], RELATION_NAMES, False),
        (models[1], winner['guardian_threshold'], paths[1], ACOUSTIC_NAMES, True),
    ):
        export(model, threshold, path, [{'x': features(item, item['pairs'], acoustic=acoustic)}
                                       for item in items if len(item['pairs'])],
               feature_names=names, feature_version=VERSION)
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(items), 'boundaries': sum(len(item['pairs']) for item in items),
              'sha256': {path.name: digest(path) for path in paths},
              'scope': 'All validation boundaries: frozen sklearn/portable probabilities agree at 1e-12; arrays only, no routing change.'}
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    run(args.source, args.fresh)
