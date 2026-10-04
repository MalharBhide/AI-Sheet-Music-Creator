"""Export gated V8 models as arrays; verify every validation prediction."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from app.services.left_baseline_evidence import (
    ACOUSTIC_EVIDENCE_NAMES,
    EVIDENCE_VERSION,
    RELATION_EVIDENCE_NAMES,
)
from export_context_verifier import export
from note_relations import ALL_NAMES
from train_left_consensus_v8 import acoustic_view, load_candidate, prepare
from train_note_verifier import load_data, write
from train_residual_verifier import cached_external


def validation_items(directory, run):
    items = load_data(directory, 'validation')
    settings = json.loads((run / 'run.json').read_text())
    for filename, digest in settings['extra_manifest_sha256'].items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen fitting/validation manifest changed')
        items.extend(item for item in cached_external(directory, path, directory) if item['group'] == 'validation')
    if len(items) != settings['validation_clips']:
        raise ValueError('Frozen validation count changed')
    return items


def release_reports(run, selection):
    selection_hash = hashlib.sha256((run / 'selection.json').read_bytes()).hexdigest()
    for stage in ('regression', 'fresh'):
        report = json.loads((run / (stage + '.json')).read_text())
        if (not report['passes'] or report['checkpoint_sha256'] != selection['checkpoint_sha256']
                or report['selection_sha256'] != selection_hash or report['threshold'] != selection['threshold']
                or report['guardian_threshold'] != selection['guardian_threshold']):
            raise ValueError('Failed or changed release evaluation')
    if json.loads((run / 'regression.json').read_text())['false_notes_removed'] <= 0:
        raise ValueError('Release must improve on website V7')


def run(directory, source):
    selection, models = load_candidate(source)
    release_reports(source, selection)
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    evidence = selection['feature_version'] == EVIDENCE_VERSION
    items = prepare(directory, validation_items(directory, source), evidence,
                    selection['policy'] == 'v7-all-accompaniment-consensus')
    relation, guardian = output / 'left-relations-v8.npz', output / 'left-guardian-v8.npz'
    export(models[0], selection['threshold'], relation, items,
           feature_names=RELATION_EVIDENCE_NAMES if evidence else ALL_NAMES, feature_version=selection['feature_version'])
    contract = {'feature_names': ACOUSTIC_EVIDENCE_NAMES, 'feature_version': EVIDENCE_VERSION} if evidence else {}
    export(models[1], selection['guardian_threshold'], guardian, [{'x': acoustic_view(item['x'])} for item in items], **contract)
    result = {'passes': True, 'source_sha256': selection['checkpoint_sha256'],
              'selection_sha256': hashlib.sha256((source / 'selection.json').read_bytes()).hexdigest(),
              'threshold': selection['threshold'], 'guardian_threshold': selection['guardian_threshold'],
              'recordings': len(items), 'candidate_events': sum(len(item['events']) for item in items),
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (relation, guardian)},
              'check': 'Portable heads match frozen sklearn predictions at rtol/atol 1e-12 on every validation candidate'}
    write(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    run(args.directory, args.source)
