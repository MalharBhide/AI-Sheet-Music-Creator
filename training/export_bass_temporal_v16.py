"""Portable V16 weights require positive fresh gain and all preservation gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from app.services.bass_temporal_refinement import RefinedTemporalCandidateModel
from app.services.temporal_note_model import VERSION, probability
from bass_temporal_v16_evidence import all_items
from prepare_robust_training_stems import digest, preserve


def export(run, fresh):
    _, winner, model, normalizer, items = all_items(run, fresh)
    output = run / 'portable'
    output.mkdir(exist_ok=False)
    saved = torch.load(run / winner['checkpoint'], map_location='cpu', weights_only=True)
    arrays = {'version': VERSION, 'format_version': 1, 'width': saved['width'],
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'mean': normalizer[0], 'scale': normalizer[1]}
    arrays.update({'weight_' + name: tensor.numpy() for name, tensor in model.state_dict().items()})
    path = output / 'bass-temporal-refinement-v1.npz'
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        portable = RefinedTemporalCandidateModel(saved)
    count, error = 0, 0.
    for item in items:
        expected = probability(model, item['frames'], item['x'], normalizer)
        actual = portable.probability(item['frames'], item['x'])
        np.testing.assert_array_equal(expected, actual, err_msg=item['id'])
        count += len(expected)
        error = max(error, float(np.max(np.abs(expected-actual), initial=0.)))
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(run / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(items), 'v15_retained_events': count,
              'sha256': {path.name: digest(path)}, 'maximum_probability_error': error,
              'scope': 'Every cached V15 survivor on all fitting, validation and consumed regressions including the first pass; portable probability parity only, no production routing.'}
    preserve(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    args = parser.parse_args()
    export(args.run, args.fresh)
