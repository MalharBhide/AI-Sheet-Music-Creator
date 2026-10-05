"""Extract fixed note-centred sequences from sealed V14 dataset audio only."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.temporal_note_model import HOP, OFFSETS, RATE, STEPS, VERSION, sequences, spectrum
from bass_training_data import identity
from current_bass_v14_baseline import hashes
from prepare_robust_training_stems import digest, preserve


def prepare(parent, output):
    plan = json.loads((parent / 'plan.json').read_text())
    manifest = json.loads((parent / 'manifest.json').read_text())
    if (plan['baseline_hashes'] != hashes() or manifest['plan_sha256'] != digest(parent / 'plan.json')
            or len(manifest['items']) != 897 or not manifest['all_tests_consumed_regression']
            or not manifest['no_user_audio_or_scores']):
        raise ValueError('Changed current V14 source baseline')
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    names = ('training/prepare_bass_temporal.py', 'backend/app/services/temporal_note_model.py',
             'backend/app/services/bass_harmonic.py', 'backend/app/services/note_evidence.py')
    prepared = {'version': VERSION, 'baseline_hashes': hashes(), 'parent_root': str(parent),
                'parent_plan_sha256': digest(parent / 'plan.json'), 'parent_manifest_sha256': digest(parent / 'manifest.json'),
                'code_sha256': {name: digest(root / name) for name in names}, 'rate': RATE, 'hop': HOP,
                'steps': STEPS, 'pitch_offsets': list(OFFSETS), 'groups': ['train', 'validation'],
                'no_user_audio_or_scores': True, 'no_test_audio_or_metrics': True,
                'scope': 'Note-centred CQT envelopes on actual V14 survivors; training/validation only. Labels and clocks unchanged. Original/GuitarSet/BabySlakh audio, no user uploads.'}
    preserve(output / 'plan.json', prepared)
    work = output / 'features'
    work.mkdir()
    records = []
    for record in manifest['items']:
        if record['group'] not in prepared['groups']:
            continue
        source = parent / 'features' / (record['id'] + '.npz')
        if digest(source) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
            raise ValueError('Changed temporal source identity')
        with np.load(source, allow_pickle=False) as saved:
            if (str(saved['plan_sha256']) != manifest['plan_sha256']
                    or str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})):
                raise ValueError('Changed source note identity')
            events = saved['events']
        samples, rate = sf.read(record['audio'], dtype='float32')
        if len(samples) / rate != record['duration']:
            raise ValueError('Changed temporal audio clock')
        frames = sequences(spectrum(samples, rate), events)
        row = {key: record[key] for key in ('id', 'group', 'source_group', 'audio', 'audio_sha256')}
        row.update(source_cache_sha256=record['cache_sha256'], notes=len(events))
        destination = work / (row['id'] + '.npz')
        np.savez_compressed(destination, frames=frames, events=events, identity=identity(row),
                            plan_sha256=digest(output / 'plan.json'))
        row['cache_sha256'] = digest(destination)
        records.append(row)
        if len(records) % 25 == 0:
            print(json.dumps({'cached': len(records), 'id': row['id'], 'notes': len(events)}), flush=True)
    if len(records) != 677 or prepared['code_sha256'] != {name: digest(root / name) for name in names}:
        raise ValueError('Incomplete temporal preparation or changed producer')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_test_audio_or_metrics': True, 'no_user_audio_or_scores': True})
    print(json.dumps({'completed': len(records), 'notes': sum(r['notes'] for r in records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.output)
