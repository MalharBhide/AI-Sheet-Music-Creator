"""Export frozen, gated consensus heads as arrays with exact prediction parity."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch
from export_context_verifier import export
from note_relations import ALL_NAMES, RELATION_VERSION
from train_left_relations_v7 import hashes, prepare
from train_note_verifier import load_data, write
from train_residual_verifier import cached_external


def run(directory, source):
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    selection = json.loads((source / 'selection.json').read_text())
    if selection['baseline_hashes'] != hashes() or not selection['selected']:
        raise ValueError('Invalid frozen selection')
    for stage in ('regression', 'fresh'):
        report = json.loads((source / (stage + '.json')).read_text())
        if not report['passes'] or report['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('Release gate failed or checkpoint changed')
    checkpoint = source / 'candidate.pickle'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Trained candidate changed')
    with checkpoint.open('rb') as stream:
        models = pickle.load(stream)
    if [width for width, _ in models] != [72, 52]:
        raise ValueError('Unexpected consensus feature contracts')
    items = load_data(directory, 'validation')
    for name in ('relational-piano-training', 'left-hand-expanded-data'):
        items.extend(item for item in cached_external(directory, directory.parent / name / 'manifest.json', directory)
                     if item['group'] == 'validation')
    torch.set_num_threads(2)
    items = prepare(directory, items)
    relation = output / 'left-relations-v7.npz'
    guardian = output / 'left-guardian-v7.npz'
    export(models[0][1], selection['threshold'], relation, items,
           feature_names=ALL_NAMES, feature_version=RELATION_VERSION)
    export(models[1][1], selection['guardian_threshold'], guardian,
           [{'x': item['x'][:, :52]} for item in items])
    result = {'passes': True, 'source_sha256': selection['checkpoint_sha256'],
              'threshold': selection['threshold'], 'guardian_threshold': selection['guardian_threshold'],
              'recordings': len(items), 'candidate_events': sum(len(item['events']) for item in items),
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (relation, guardian)},
              'check': 'Each portable head matches sklearn on every validation feature at rtol/atol 1e-12'}
    write(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    run(args.directory, args.source)
