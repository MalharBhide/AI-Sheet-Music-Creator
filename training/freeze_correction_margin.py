"""Freeze the documented conservative follow-up, without another grid search."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from evaluate_context_correction import evaluate
from train_note_verifier import load_data, write


def freeze(directory):
    source, output = directory / 'agreement-correction', directory / 'agreement-margin'
    output.mkdir(exist_ok=False)
    hashes = []
    for index in range(2):
        with np.load(source / f'candidate-{index}.npz', allow_pickle=False) as arrays:
            saved = dict(arrays)
        saved['threshold'] = np.asarray(.01)
        path = output / f'candidate-{index}.npz'
        np.savez_compressed(path, **saved)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    write(output / 'selection.json', {'checkpoint_sha256': hashes, 'prune': .01,
        'ceiling': .2, 'rescue': 1.01, 'test_not_evaluated': True,
        'mode': 'Post-failure development safety margin: 5x lower score threshold than validation-selected .05; both heads agree and V2 score <= .2. No further parameter search. All 118 prior evaluation excerpts are consumed.',
        'final_check': 'Oxford 60–90s, excluding previously scored and clock-calibration notes; same recordings, not independent performers'})
    write(output / 'validation.json', evaluate(directory, output, load_data(directory, 'validation')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    freeze(parser.parse_args().directory)
