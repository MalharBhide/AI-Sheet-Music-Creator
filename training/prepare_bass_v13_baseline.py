"""Seal V13 survivors from the completed release, without redoing inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_texture import BassTexture
from bass_training_data import eligible, identity
from current_bass_v13_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

WITNESSES = {
    'plan.json': '6b5a2524f3adf4dae2a7aeacccc5490e8e046e8490297b4523273c0354908b22',
    'runtime-parity.json': '305fad1bd4a2f46d7a3583e9da53e935a8d49d7df1977735960fa4625dda5d68',
    'native-routing.json': '2620a5186d7bbbbcfce1310db061a6fadee378137f701a99626a252544b27603',
    'original-first-pass.json': 'b0462acff307659e2b312af4c24885452b53e6d74317ca1b265b7273e1d7a395',
    'release-seal.json': '77825687a2b9a1aa5fb2ce72e2807cc79e2ef8ac0fb39063a082a0bde8231cbb',
}
COUNTS = {'train': 531, 'validation': 146, 'test': 188}


def sealed_sources(run, fresh):
    hashes()
    for name, expected in WITNESSES.items():
        if digest(run / name) != expected:
            raise ValueError('Changed sealed V13 release evidence')
    seal = json.loads((run / 'release-seal.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in seal['sources'].items():
        if name != 'backend/app/services/piano_transcription.py' and digest(root / name) != expected:
            raise ValueError('Changed historical evidence source')
    for name, expected in seal['witnesses'].items():
        if digest(run / name) != expected:
            raise ValueError('Changed completed V13 release witness')
    plan = json.loads((run / 'plan.json').read_text())
    texture = Path(plan['baseline_cache_root'])
    texture_plan = json.loads((texture / 'plan.json').read_text())
    parent = Path(texture_plan['parent_cache_root'])
    earlier = Path(texture_plan['consumed_first_pass_root'])
    collections = [(parent, texture_plan['parent_cache_manifest_sha256'], texture_plan['parent_cache_plan_sha256'], True),
                   (earlier, texture_plan['consumed_first_pass_manifest_sha256'], None, False),
                   (texture, plan['baseline_cache_manifest_sha256'], plan['baseline_cache_plan_sha256'], False),
                   (fresh, seal['fresh_manifest_sha256'], None, False)]
    rows = []
    for folder, manifest_sha, plan_sha, parent_cache in collections:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if (digest(folder / 'manifest.json') != manifest_sha
                or manifest['plan_sha256'] != digest(folder / 'plan.json')
                or (plan_sha and digest(folder / 'plan.json') != plan_sha)):
            raise ValueError('Changed V13 source manifest or plan')
        if folder in (earlier, fresh) and (not manifest['now_consumed_regression']
                or any(r['group'] != 'test' for r in manifest['items'])):
            raise ValueError('Consumed tests cannot become fitting or selection')
        for record in manifest['items']:
            path = (folder / 'features' / (record['id'] + '.npz')) if parent_cache else folder / record['id'] / 'features.npz'
            rows.append((record, path, manifest['plan_sha256'], parent_cache or folder == texture))
    groups = {g: {r['source_group'] for r, *_ in rows if r['group'] == g} for g in COUNTS}
    if (len({r['id'] for r, *_ in rows}) != 865
            or {g: sum(r['group'] == g for r, *_ in rows) for g in COUNTS} != COUNTS
            or any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))):
        raise ValueError('V13 source partitions changed or leaked')
    return rows


def source(record, path, plan_sha, has_identity):
    if digest(path) != record['cache_sha256'] or digest(Path(record['audio'])) != record['audio_sha256']:
        raise ValueError('Changed V13 source audio or interval cache')
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['plan_sha256']) != plan_sha or (has_identity and
                str(saved['identity']) != identity({k: v for k, v in record.items() if k != 'cache_sha256'})):
            raise ValueError('Changed V13 source cache identity')
        item = {**record, **{key: saved[key] for key in ('base_x', 'events', 'eligible')}}
    if sf.info(item['audio']).duration != item['duration']:
        raise ValueError('Changed V13 audio clock')
    np.testing.assert_array_equal(item['eligible'], eligible(item['events'], item['duration']))
    return item


def prepare(run, fresh, output):
    rows = sealed_sources(run, fresh)
    output.mkdir(exist_ok=False)
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'release_root': str(run),
            'witness_sha256': WITNESSES, 'consumed_first_pass_root': str(fresh),
            'code_sha256': digest(Path(__file__)), 'expected_counts': COUNTS,
            'no_audio_inference': True, 'no_user_audio_or_scores': True,
            'scope': 'All 865 sealed V12 recordings, apply released V13 deletion only; recompute neighbors. 188 consumed tests never fit/select.'}
    preserve(output / 'plan.json', plan)
    work = output / 'features'
    work.mkdir()
    texture, records = BassTexture(), []
    for record, path, plan_sha, has_identity in rows:
        item = source(record, path, plan_sha, has_identity)
        result = decisions(item, item['duration'], texture)
        row = {key: record[key] for key in ('id', 'group', 'source_group', 'corpus', 'audio', 'seconds',
               'reference', 'pitch_reference', 'duration', 'audio_sha256')}
        row.update(source_cache_sha256=record['cache_sha256'], retained_v12=len(item['events']),
                   retained_v13=len(result['events']), texture_rejections=result['texture_rejections'])
        destination = work / (row['id'] + '.npz')
        np.savez_compressed(destination, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                            identity=identity(row), plan_sha256=digest(output / 'plan.json'))
        row['cache_sha256'] = digest(destination)
        records.append(row)
    count = sum(r['retained_v12'] for r in records)
    removed = sum(r['texture_rejections'] for r in records)
    if count != 48113 or removed != 315:
        raise ValueError('V13 decisions differ from completed runtime parity')
    if hashes() != plan['baseline_hashes']:
        raise ValueError('V13 changed during baseline preparation')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_user_audio_or_scores': True, 'all_tests_consumed_regression': True})
    evidence = {'passes': True, 'recordings': len(records), 'v12_retained_events': count,
                'v13_rejections': removed, 'v13_retained_events': count - removed, 'groups': COUNTS,
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
