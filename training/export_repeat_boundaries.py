"""Export boundary arrays after frozen regression, current-piano and stress gates."""

import argparse
import json
from pathlib import Path

import torch
from app.services.repeat_boundary import ACOUSTIC_NAMES, RELATION_NAMES, VERSION
from current_accompaniment_baseline import prepare
from export_context_verifier import export
from repeat_boundary_features import features
from repeat_boundary_release import load_frozen, require_report
from train_broad_consensus import digest, validation_items
from train_note_verifier import write
from train_repeat_boundaries import annotate


def run(directory, source):
    winner, models = load_frozen(source)
    require_report(source, winner, 'regression', positive=True)
    require_report(source, winner, 'current-piano-regression')
    require_report(source, winner, 'fresh-stress')
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    items = annotate(prepare(directory, validation_items(directory, source)))
    relation, guardian = output / 'repeat-boundary-v1.npz', output / 'repeat-boundary-guardian-v1.npz'
    for model, threshold, path, names, acoustic in (
        (models[0], winner['threshold'], relation, RELATION_NAMES, False),
        (models[1], winner['guardian_threshold'], guardian, ACOUSTIC_NAMES, True),
    ):
        export(model, threshold, path, [{'x': features(item, item['pairs'], acoustic=acoustic)}
                                       for item in items if len(item['pairs'])],
               feature_names=names, feature_version=VERSION)
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(items), 'boundaries': sum(len(item['pairs']) for item in items),
              'sha256': {path.name: digest(path) for path in (relation, guardian)},
              'scope': 'Every validation boundary prediction agrees with frozen sklearn at 1e-12; arrays only'}
    write(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    run(args.directory, args.source)
