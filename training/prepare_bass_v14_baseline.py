"""Seal V14 survivors from the completed release, without redoing inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import BassHarmonic
from bass_training_data import eligible, identity
from current_bass_v14_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

WITNESSES = {'plan.json': '55b492a158f37ed430bad4df01ec765e810942b46544148fc49c9cf280ab5f9a', 'runtime-parity.json': 'e2cce7e0207522794fecde85a4d3cbb6fb6b1b4860b0c3bf5aacfcf59093db7b', 'native-routing.json': 'e8ebd474eb035aefd05aa46fab5a61b33c9bdca98816a2cf49d4b00c4e523fa9', 'original-first-pass.json': '6162212f6881ba3cb841e6816629aa25ab2f7ffadf3dc517a070eb4ae4687635', 'release-seal.json': 'c753da1175def7692aeeae9458f59d207180bf777bb14569567082a73bfa705a'}

COUNTS = {'train': 531, 'validation': 146, 'test': 220}


def sealed_sources(run, fresh):
    hashes()
    for name, expected in WITNESSES.items():
        if digest(run / name) != expected:
            raise ValueError('Changed sealed V14 release evidence')
    seal = json.loads((run / 'release-seal.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in seal['sources'].items():
        if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
            raise ValueError('Changed historical evidence source')
    for name, expected in seal['witnesses'].items():
        if digest(run / name) != expected:
            raise ValueError('Changed completed V14 release witness')
    plan = json.loads((run / 'plan.json').read_text())
    parent = Path(plan['baseline_cache_root'])
    collections = [(parent, plan['baseline_cache_manifest_sha256'], plan['baseline_cache_plan_sha256'], True),
                   (fresh, seal['fresh_manifest_sha256'], None, False)]
    rows = []
    for folder, manifest_sha, plan_sha, parent_cache in collections:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if (digest(folder / 'manifest.json') != manifest_sha
                or manifest['plan_sha256'] != digest(folder / 'plan.json')
                or (plan_sha and digest(folder / 'plan.json') != plan_sha)):
            raise ValueError('Changed V14 source manifest or plan')
        if folder == fresh and (not manifest['now_consumed_regression']
                or any(r['group'] != 'test' for r in manifest['items'])):
            raise ValueError('Consumed tests cannot become fitting or selection')
        for record in manifest['items']:
            path = (folder / 'features' / (record['id'] + '.npz')) if parent_cache else folder / record['id'] / 'features.npz'
            rows.append((record, path, manifest['plan_sha256'], parent_cache))
    groups = {g: {r['source_group'] for r, *_ in rows if r['group'] == g} for g in COUNTS}
    if (len({r['id'] for r, *_ in rows}) != 897
            or {g: sum(r['group'] == g for r, *_ in rows) for g in COUNTS} != COUNTS
            or any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))):
        raise ValueError('V14 source partitions changed or leaked')
    return rows


def source(record, path, plan_sha, has_identity):
    if digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
        raise ValueError('Changed V14 source audio or interval cache')
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['plan_sha256']) != plan_sha or (has_identity and
                str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})):
            raise ValueError('Changed V14 source cache identity')
        item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible')}}
    if sf.info(item['audio']).duration != item['duration']:
        raise ValueError('Changed V14 audio clock')
    np.testing.assert_array_equal(item['eligible'], eligible(item['events'], item['duration']))
    return item


def prepare(run, fresh, output):
    rows = sealed_sources(run, fresh)
    output.mkdir(exist_ok=False)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'release_root': str(run),
            'witness_sha256': WITNESSES, 'consumed_first_pass_root': str(fresh),
            'code_sha256': digest(Path(__file__)), 'expected_counts': COUNTS,
            'no_audio_inference': True, 'no_user_audio_or_scores': True,
            'scope': 'All 897 sealed V13 recordings, apply released V14 deletion only; recompute neighbors. 220 consumed tests never fit/select.'}
    preserve(output / 'plan.json', plan)
    work = output / 'features'
    work.mkdir()
    harmonic, records = BassHarmonic(), []
    for record, path, plan_sha, has_identity in rows:
        item = source(record, path, plan_sha, has_identity)
        result = decisions(item, item['duration'], harmonic)
        row = {key: record[key] for key in ('id', 'group', 'source_group', 'corpus', 'audio', 'seconds',
               'reference', 'pitch_reference', 'duration', 'audio_sha256')}
        row.update(source_cache_sha256=record['cache_sha256'], retained_v13=len(item['events']),
                   retained_v14=len(result['events']), harmonic_rejections=result['harmonic_rejections'])
        destination = work / (row['id'] + '.npz')
        np.savez_compressed(destination, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                            identity=identity(row), plan_sha256=digest(output / 'plan.json'))
        row['cache_sha256'] = digest(destination)
        records.append(row)
    count = sum(r['retained_v13'] for r in records)
    removed = sum(r['harmonic_rejections'] for r in records)
    if count != 48621 or removed != 174:
        raise ValueError('V14 decisions differ from completed runtime parity')
    if hashes() != plan['baseline_hashes']:
        raise ValueError('V14 changed during baseline preparation')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_user_audio_or_scores': True, 'all_tests_consumed_regression': True})
    evidence = {'passes': True, 'recordings': len(records), 'v13_retained_events': count,
                'v14_rejections': removed, 'v14_retained_events': count - removed, 'groups': COUNTS,
                'no_audio_inference': True, 'plan_sha256': digest(output / 'plan.json'),
                'manifest_sha256': digest(output / 'manifest.json')}
    preserve(output / 'preparation-evidence.json', evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fresh', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.run, args.fresh, args.output)
