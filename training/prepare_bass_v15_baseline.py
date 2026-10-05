"""Seal actual V15 survivors using completed release caches, without inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_temporal import BassTemporal
from bass_training_data import eligible, identity
from current_bass_v15_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

COUNTS = {'train': 531, 'validation': 146, 'test': 252}
WITNESSES = {
    'runtime-parity.json': '9070c087e23a2b7d502e1d13c9e6979594e0977df3bf94519cecfb59556d4084',
    'release-seal.json': '323466a48197a565e2729336e4fa25c2cd2dda8ddf2a8a4eb398078d229b7977',
}


def read_manifest(folder, expected):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(folder / 'manifest.json') != expected or manifest['plan_sha256'] != digest(folder / 'plan.json'):
        raise ValueError('Changed sealed V15 source manifest')
    return manifest


def sources(run, regression, fresh):
    hashes()
    for name, expected in WITNESSES.items():
        if digest(run / name) != expected:
            raise ValueError('Changed completed V15 release witness')
    seal = json.loads((run / 'release-seal.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in seal['sources'].items():
        if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
            raise ValueError('Changed completed V15 evidence source')
    for name, expected in seal['witnesses'].items():
        if digest(run / name) != expected:
            raise ValueError('Changed completed V15 release report')
    native = json.loads((run / 'native-routing.json').read_text())
    if not native['passes'] or native['routing_source_sha256'] != hashes()['sources']['backend/app/services/piano_transcription.py']:
        raise ValueError('Unverified V15 native route')
    plan = json.loads((run / 'plan.json').read_text())
    sequence = Path(plan['sequence_root'])
    seq_plan = json.loads((sequence / 'plan.json').read_text())
    parent = Path(seq_plan['parent_root'])
    base = read_manifest(parent, seq_plan['parent_manifest_sha256'])
    seq = read_manifest(sequence, plan['sequence_manifest_sha256'])
    consumed = json.loads((run / 'consumed-regression.json').read_text())
    reg = read_manifest(regression, consumed['regression_manifest_sha256'])
    fresh_manifest = read_manifest(fresh, seal['fresh_manifest_sha256'])
    if (not fresh_manifest['now_consumed_regression'] or not base['all_tests_consumed_regression']
            or not seq['no_test_audio_or_metrics'] or not reg['no_fitting_or_selection']):
        raise ValueError('Changed V15 partition policy')
    frames = {}
    for folder, manifest in ((sequence, seq), (regression, reg)):
        for record in manifest['items']:
            path = folder / 'features' / (record['id'] + '.npz')
            if record['id'] in frames:
                raise ValueError('Duplicate V15 frame source')
            frames[record['id']] = (record, path, manifest['plan_sha256'])
    rows = []
    for record in base['items']:
        frame_record, frame_path, frame_plan = frames.pop(record['id'])
        if (frame_record['source_cache_sha256'] != record['cache_sha256']
                or frame_record['source_group'] != record['source_group'] or frame_record['group'] != record['group']):
            raise ValueError('Changed V15 paired spectral source')
        rows.append((record, parent / 'features' / (record['id'] + '.npz'), base['plan_sha256'],
                     frame_record, frame_path, frame_plan))
    if frames:
        raise ValueError('Extra V15 spectral sources')
    for record in fresh_manifest['items']:
        if record['group'] != 'test':
            raise ValueError('Consumed V15 first-pass source cannot fit or select')
        path = fresh / record['id'] / 'features.npz'
        rows.append((record, path, fresh_manifest['plan_sha256'], None, path, fresh_manifest['plan_sha256']))
    groups = {g: {r['source_group'] for r, *_ in rows if r['group'] == g} for g in COUNTS}
    if (len({r['id'] for r, *_ in rows}) != 929
            or {g: sum(r['group'] == g for r, *_ in rows) for g in COUNTS} != COUNTS
            or any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))):
        raise ValueError('V15 source partitions changed or leaked')
    return rows


def source(row):
    record, path, plan_sha, frame_record, frame_path, frame_plan = row
    if digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
        raise ValueError('Changed V15 source audio or intervals')
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['plan_sha256']) != plan_sha or (frame_record and
                str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})):
            raise ValueError('Changed V15 source cache identity')
        item = {**record, **{key: saved[key] for key in ('events', 'base_x', 'eligible')}}
    if frame_record and digest(frame_path) != frame_record['cache_sha256']:
        raise ValueError('Changed V15 spectral cache')
    with np.load(frame_path, allow_pickle=False) as saved:
        if str(saved['plan_sha256']) != frame_plan or (frame_record and
                str(saved['identity']) != identity({k: v for k, v in frame_record.items() if k != 'cache_sha256'})):
            raise ValueError('Changed V15 spectral cache identity')
        np.testing.assert_array_equal(saved['events'], item['events'])
        item['frames'] = saved['frames']
    if sf.info(item['audio']).duration != item['duration']:
        raise ValueError('Changed V15 source clock')
    np.testing.assert_array_equal(item['eligible'], eligible(item['events'], item['duration']))
    return item


def prepare(run, regression, fresh, output):
    rows = sources(run, regression, fresh)
    output.mkdir(exist_ok=False)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'release_root': str(run),
            'witness_sha256': WITNESSES, 'code_sha256': digest(Path(__file__)), 'expected_counts': COUNTS,
            'no_audio_inference': True, 'no_user_audio_or_scores': True,
            'scope': 'All 929 completed V15 recordings; reuse per-note CQT evidence, apply released temporal deletion only. 252 consumed tests never fit/select.'}
    preserve(output / 'plan.json', plan)
    work = output / 'features'
    work.mkdir()
    temporal, records = BassTemporal(), []
    for row in rows:
        item = source(row)
        result = decisions(item, item['duration'], temporal)
        record = {key: item[key] for key in ('id', 'group', 'source_group', 'corpus', 'audio', 'seconds',
                  'reference', 'pitch_reference', 'duration', 'audio_sha256')}
        record.update(source_cache_sha256=item['cache_sha256'], retained_v14=len(item['events']),
                      retained_v15=len(result['events']), temporal_rejections=result['temporal_rejections'])
        path = work / (record['id'] + '.npz')
        np.savez_compressed(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                            identity=identity(record), plan_sha256=digest(output / 'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
    count, removed = sum(r['retained_v14'] for r in records), sum(r['temporal_rejections'] for r in records)
    if count != 49214 or removed != 74 or plan['baseline_hashes'] != hashes():
        raise ValueError('V15 survivor decisions changed from completed runtime witness')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'all_tests_consumed_regression': True, 'no_user_audio_or_scores': True})
    result = {'passes': True, 'recordings': len(records), 'v14_retained_events': count,
              'v15_rejections': removed, 'v15_retained_events': count-removed, 'groups': COUNTS,
              'no_audio_inference': True, 'manifest_sha256': digest(output / 'manifest.json')}
    preserve(output / 'preparation-evidence.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'regression', 'fresh', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.run, args.regression, args.fresh, args.output)
