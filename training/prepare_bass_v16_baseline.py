"""Seal V16 survivor evidence for timing/decluttering, reusing released caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_temporal_refinement import BassTemporalRefinement
from bass_temporal_v15_data import read_items
from bass_temporal_v16_evidence import cached_items
from bass_training_data import identity
from current_bass_v16_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

COUNTS = {'train':573, 'validation':160, 'test':296}
WITNESSES = {'runtime-parity.json':'b866113afa733bcd78b687ec6a98c8df73a7809abb0d86af74e772c4ec42bc63',
             'release-seal.json':'94ba1512fd7b429af7dc711edea46716aaba5e05a81fa3e910c9f95ded7ac644',
             'native-routing.json':'0824cb10d4ed14d1b5524dbdfd936798b7488ac1a77113640371ac19504f6d05'}


def sources(run):
    current = hashes()
    for name, expected in WITNESSES.items():
        if digest(run/name) != expected:
            raise ValueError('Changed completed V16 release witness')
    seal = json.loads((run/'release-seal.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in seal['sources'].items():
        if name != 'backend/app/services/piano_transcription.py' and digest(root/name) != expected:
            raise ValueError('Changed sealed V16 producer or runtime')
    for name, expected in seal['witnesses'].items():
        if digest(run/name) != expected:
            raise ValueError('Changed V16 sealed evidence')
    native = json.loads((run/'native-routing.json').read_text())
    if not native['passes'] or native['routing_source_sha256'] != current['sources']['backend/app/services/piano_transcription.py']:
        raise ValueError('Changed verified V16 native route')
    plan = json.loads((run/'plan.json').read_text())
    rows, bindings = [], []
    for key, baseline in (('baseline',True),('piano',False)):
        folder = Path(plan[key+'_root'])
        manifest = json.loads((folder/'manifest.json').read_text())
        if (digest(folder/'manifest.json') != plan[key+'_manifest_sha256']
                or manifest['plan_sha256'] != digest(folder/'plan.json')):
            raise ValueError('Changed V16 fitting source manifest')
        rows += read_items(folder,manifest['items'],manifest['plan_sha256'],('train','validation','test'),baseline)
        bindings.append({'root':str(folder),'manifest_sha256':digest(folder/'manifest.json')})
    for report, folder, key in (
        ('piano-regression.json',run.parent/'v16-piano-regression-v1','manifest_sha256'),
        ('original-first-pass.json',run.parent/'bass-temporal-v16-fresh-v1','test_manifest_sha256')):
        completed = json.loads((run/report).read_text())
        manifest = json.loads((folder/'manifest.json').read_text())
        if (not completed['passes'] or digest(folder/'manifest.json') != completed[key]
                or manifest['plan_sha256'] != digest(folder/'plan.json') or not manifest['now_consumed_regression']):
            raise ValueError('Changed V16 consumed regression evidence')
        rows += cached_items(folder,manifest)
        bindings.append({'root':str(folder),'manifest_sha256':digest(folder/'manifest.json')})
    partitions = {g:{r['source_group'] for r in rows if r['group']==g} for g in COUNTS}
    if (len({r['id'] for r in rows}) != 1029
            or {g:sum(r['group']==g for r in rows) for g in COUNTS} != COUNTS
            or any(partitions[a]&partitions[b] for a,b in (('train','validation'),('train','test'),('validation','test')))):
        raise ValueError('Incomplete, duplicated or leaking V16 source partition')
    return rows, bindings


def prepare(run,output):
    rows, bindings = sources(run)
    output.mkdir(exist_ok=False)
    plan = {'version':VERSION,'baseline_hashes':hashes(),'expected_counts':COUNTS,
            'release_root':str(run),'witness_sha256':WITNESSES,'producer_sha256':digest(Path(__file__)),
            'source_manifests':bindings,'no_audio_inference':True,'no_user_audio_or_scores':True,
            'scope':'Actual V16 survivors on all 1029 released recordings. Per-note spectral/acoustic reuse is valid because V16 only deletes notes. 296 consumed tests never fit or select.'}
    preserve(output/'plan.json',plan)
    work = output/'features'
    work.mkdir()
    verifier,records = BassTemporalRefinement(),[]
    for item in rows:
        if sf.info(item['audio']).duration != item['duration']:
            raise ValueError('Changed V16 source audio clock')
        result = decisions(item,verifier)
        record = {key:item[key] for key in ('id','group','source_group','corpus','audio','seconds','reference',
                                          'pitch_reference','duration','audio_sha256')}
        for key in ('reference','pitch_reference'):
            record[key] = np.asarray(record[key],float).tolist()
        record.update(retained_v15=len(item['events']),retained_v16=len(result['events']),
                      v16_rejections=result['v16_rejections'],source_cache_sha256=item['cache_sha256'])
        path = work/(record['id']+'.npz')
        np.savez_compressed(path,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)},
                            identity=identity(record),plan_sha256=digest(output/'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
    count,removed = sum(r['retained_v15'] for r in records),sum(r['v16_rejections'] for r in records)
    if count != 55163 or removed != 71 or plan['baseline_hashes'] != hashes():
        raise ValueError('V16 survivor decisions differ from completed runtime witness')
    preserve(output/'manifest.json',{'version':VERSION,'plan_sha256':digest(output/'plan.json'),'items':records,
                                    'all_tests_consumed_regression':True,'no_user_audio_or_scores':True})
    result = {'passes':True,'recordings':len(records),'v15_retained_events':count,'v16_rejections':removed,
              'v16_retained_events':count-removed,'groups':COUNTS,'no_audio_inference':True,
              'manifest_sha256':digest(output/'manifest.json')}
    preserve(output/'preparation-evidence.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    prepare(args.run,args.output)
