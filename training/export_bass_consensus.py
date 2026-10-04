"""Export only a frozen bass winner with positive, preservation-safe test gain."""

import argparse
import json
from pathlib import Path

from app.services.note_evidence import CONTEXT_VERSION, FEATURE_NAMES
from bass_release import load_frozen, require_report
from bass_training_data import NAMES, load
from export_context_verifier import export
from prepare_robust_training_stems import digest, preserve


def run_export(directory, run):
    winner, models = load_frozen(directory, run)
    require_report(run, winner, 'held-regression')
    require_report(run, winner, 'reserved-slakh', positive=True)
    output = run / 'portable'
    output.mkdir(exist_ok=False)
    validation = load(directory, 'validation')
    result = {'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'held_report_sha256': digest(run / 'held-regression.json'),
              'reserved_report_sha256': digest(run / 'reserved-slakh.json'),
              'export_code_sha256': digest(Path(__file__)), 'arrays': [],
              'parity_tolerance': 1e-12, 'candidate_events': sum(len(item['events']) for item in validation),
              'deployed': False, 'scope': 'Portable verification only; production integration/parity still required.'}
    for model, name, threshold, names, columns in (
            (models[0], 'bass-consensus-v1.npz', winner['threshold'], NAMES, 52),
            (models[1], 'bass-guardian-v1.npz', winner['guardian_threshold'], FEATURE_NAMES, 26)):
        path = output / name
        export(model, threshold, path, [{'x': item['x'][:, :columns]} for item in validation],
               feature_names=names, feature_version=CONTEXT_VERSION)
        result['arrays'].append({'filename': name, 'sha256': digest(path), 'threshold': threshold,
                                 'feature_count': columns})
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run_export(args.directory, args.run)
