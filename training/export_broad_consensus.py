"""Export a pitch-preserving broad candidate only after frozen release gates."""

import argparse
import json
from pathlib import Path

import torch
from app.services.left_baseline_evidence import (
    ACOUSTIC_EVIDENCE_NAMES,
    EVIDENCE_VERSION,
    RELATION_EVIDENCE_NAMES,
)
from export_context_verifier import export
from train_broad_consensus import digest, load_winner, validation_items
from train_left_consensus_v8 import acoustic_view, prepare
from train_note_verifier import write


def release_reports(source, winner):
    batch_hash = digest(source / 'batch-selection.json')
    selection_hash = digest(source / winner['name'] / 'selection.json')
    plan = json.loads((source / 'plan.json').read_text())
    if plan.get('preserve_reference_pitch_coverage') is not True:
        raise ValueError('Release requires reference coverage during validation selection')
    for stage in ('regression', 'fresh'):
        report = json.loads((source / (stage + '.json')).read_text())
        if (not report['passes'] or report['checkpoint_sha256'] != winner['checkpoint_sha256']
                or report['batch_selection_sha256'] != batch_hash
                or report['selection_sha256'] != selection_hash
                or report['threshold'] != winner['threshold']
                or report['guardian_threshold'] != winner['guardian_threshold']
                or any(not row['passes'] or row['reference_pitch_coverage_preserved'] is not True
                       for row in report['per_recording'])):
            raise ValueError('Failed or changed frozen release evaluation')
    if json.loads((source / 'regression.json').read_text())['false_notes_removed'] <= 0:
        raise ValueError('Release must improve the frozen website baseline')
    coverage = json.loads((source / 'coverage-release-gate.json').read_text())
    if (not coverage['passes'] or coverage['checkpoint_sha256'] != winner['checkpoint_sha256']
            or coverage['batch_selection_sha256'] != batch_hash
            or coverage['gate_code_sha256'] != digest(Path(__file__).with_name('check_reference_coverage.py'))
            or coverage['coverage_code_sha256'] != digest(Path(__file__).with_name('pitch_interval_coverage.py'))
            or any(not row['passes'] for row in coverage['per_recording'])):
        raise ValueError('Failed or changed reference coverage gate')


def run(directory, source):
    winner, models = load_winner(source)
    release_reports(source, winner)
    output = source / 'portable'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    raw = validation_items(directory, source)
    settings = json.loads((source / winner['name'] / 'run.json').read_text())
    if len(raw) != settings['validation_clips']:
        raise ValueError('Frozen validation cohort changed')
    items = prepare(directory, raw, baseline_evidence=True, full_register=True)
    relation, guardian = output / 'accompaniment-broad-pitch-v1.npz', output / 'accompaniment-broad-guardian-v1.npz'
    export(models[0], winner['threshold'], relation, items,
           feature_names=RELATION_EVIDENCE_NAMES, feature_version=EVIDENCE_VERSION)
    export(models[1], winner['guardian_threshold'], guardian,
           [{'x': acoustic_view(item['x'])} for item in items],
           feature_names=ACOUSTIC_EVIDENCE_NAMES, feature_version=EVIDENCE_VERSION)
    result = {'passes': True, 'checkpoint_sha256': winner['checkpoint_sha256'],
              'batch_selection_sha256': digest(source / 'batch-selection.json'),
              'threshold': winner['threshold'], 'guardian_threshold': winner['guardian_threshold'],
              'recordings': len(items), 'candidate_events': sum(len(item['events']) for item in items),
              'sha256': {p.name: digest(p) for p in (relation, guardian)},
              'scope': 'Every validation prediction matches frozen sklearn at 1e-12; arrays only, no deployed pickle'}
    write(output / 'export.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    run(args.directory, args.source)
