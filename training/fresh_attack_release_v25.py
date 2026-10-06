"""One frozen new-MP3 decluttering check; no user audio or test-driven changes."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import BassHarmonic, features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences, spectrum
from attack_local_features import observe
from attack_local_features import sequences as attack_sequences
from attack_release_model import probabilities
from bass_joint_timing_v18 import margin
from bass_training_data import eligible
from evaluate_attack_release_v25 import frozen
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from score_declutter_local import evaluate
from train_attack_release_v25 import FRESH_SEEDS


def require_regression(run, winner):
    report = json.loads((run/'consumed-regression.json').read_text())
    if (not report['passes'] or report['recordings']!=360 or not report['consumed_regression']
            or report['checkpoint_sha256']!=winner['checkpoint_sha256']
            or report['batch_selection_sha256']!=digest(run/'batch-selection.json')
            or report['test_used_for_selection'] or not report['no_user_audio_or_scores']
            or report['release_observation_manifest_sha256']!=digest(run/'consumed-release-features'/'manifest.json')):
        raise ValueError('Missing safe complete consumed timing regression')
    for name,t,g in (('raw',winner['threshold'],winner['guardian_threshold']),
                     ('margin',margin(winner['threshold']),winner['guardian_threshold'])):
        evidence = report[name]
        if (not evidence['passes'] or len(evidence['per_recording'])!=360
                or any(not row['passes'] for row in evidence['per_recording'])
                or evidence['false_notes_removed']<=0
                or (evidence['remove_threshold'],evidence['guardian_threshold'])!=(t,g)):
            raise ValueError('Changed consumed timing regression result')
    return digest(run/'consumed-regression.json')


def verify(run, output):
    report_path = run/'original-first-pass.json'
    if report_path.exists():
        raise ValueError('Preserve first-pass timing results; no replacement test')
    plan,winner,network,normalizer = frozen(run)
    if plan['fresh_seeds']!=list(FRESH_SEEDS) or plan['fresh_families']!={'original':16,'texture':16}:
        raise ValueError('Changed declared first-pass fixtures')
    regression = require_regression(run,winner)
    fine_plan = json.loads((Path(plan['baseline_root'])/'plan.json').read_text())
    consumed = json.loads((Path(fine_plan['parent_root'])/'manifest.json').read_text())
    previous=[r for source in plan['previous_fresh'] for r in json.loads((Path(source['root'])/'manifest.json').read_text())['items']]
    if ({r['source_group'] for r in consumed['items']} | {r['source_group'] for r in previous}) & {'original-bass-seed-'+str(s) for s in FRESH_SEEDS}:
        raise ValueError('First-pass fixtures overlap fitting/selection/regression sources')
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    names = ('training/fresh_attack_release_v25.py','training/prepare_bass_positive_data.py',
             'training/prepare_bass_texture_v12.py','training/prepare_slakh_bass.py',
             'backend/app/services/piano_transcription.py','backend/app/services/note_evidence.py',
             'backend/app/services/temporal_note_model.py','backend/app/services/bass_harmonic.py',
             'training/attack_local_features.py','training/attack_local_model.py',
             'training/attack_release_model.py','training/score_declutter_local.py','training/release_local_features.py')
    fixture_plan = {'seeds':list(FRESH_SEEDS),'checkpoint_sha256':winner['checkpoint_sha256'],
                    'batch_selection_sha256':digest(run/'batch-selection.json'),
                    'consumed_regression_sha256':regression,'baseline_hashes':plan['baseline_hashes'],
                    'code_sha256':{n:digest(root/n) for n in names},'rate':RATE,
                    'mp3_bitrate_kbps_cycle':[48,128,192],
                    'rights':'Project-authored procedural signals and note sequences',
                    'no_fitting_selection_user_audio_or_scores':True,
                    'scope':'32 unseen seeds of the existing original/texture generators. Actual native V16 bass predictions. No independent human-song accuracy claim.'}
    preserve(output/'plan.json',fixture_plan)
    engine,evidence,records,items = _GeneralEngine('balanced'),{},[],[]
    native = engine.predict_function

    def capture(*args,**kwargs):
        result = native(*args,**kwargs)
        evidence['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    for index,seed in enumerate(FRESH_SEEDS):
        name = 'attack-release-v25-fresh-'+str(seed)
        work = output/name
        work.mkdir()
        samples,labels = (original if index<16 else texture)(seed)
        _,clock = encode(samples,RATE,work,fixture_plan['mp3_bitrate_kbps_cycle'][index%3])
        audio = work/'decoded.wav'
        samples,rate = sf.read(audio,dtype='float32')
        if rate!=RATE or samples.ndim!=1 or len(samples)!=30*RATE:
            raise ValueError('Changed first-pass audio clock')
        with open(os.devnull,'w') as quiet, contextlib.redirect_stdout(quiet):
            midi = engine.predict(audio,'bass',90.)
            notes = [note for part in midi.instruments for note in part.notes]
            base_x = note_features(samples,rate,evidence['acoustic'],notes,include_context=True)
        events = np.asarray([[n.start,n.end,n.pitch,n.velocity] for n in notes],float).reshape(-1,4)
        x = features(events,base_x)
        frames = sequences(spectrum(samples,rate),events)
        observed = observe(samples,rate)
        attack_frames = attack_sequences(observed,events)
        release_frames = release_sequences(observed,events)
        selected = eligible(events,30.)
        record = {'id':name,'group':'test','source_group':'original-bass-seed-'+str(seed),
                  'corpus':'original-attack-release-v25-first-pass','audio':str(audio),'audio_sha256':digest(audio),
                  'duration':30.,'seconds':29.5,'reference':labels.tolist(),'pitch_reference':labels.tolist(),
                  'codec_clock':clock}
        path = work/'features.npz'
        np.savez_compressed(path,release_frames=release_frames,attack_frames=attack_frames,frames=frames,x=x,base_x=base_x,events=events,eligible=selected,
                            plan_sha256=digest(output/'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
        items.append({**record,'release_frames':release_frames,'attack_frames':attack_frames,'events':events,'x':x,'frames':frames,'base_x':base_x,'eligible':selected,
                      'reference':labels.copy(),'pitch_reference':labels.copy()})
        print(json.dumps({'cached':len(records),'id':name,'native_v16_notes':len(events)}),flush=True)
    if fixture_plan['code_sha256']!={n:digest(root/n) for n in names}:
        raise ValueError('First-pass producer changed during inference')
    frozen(run)
    require_regression(run,winner)
    preserve(output/'manifest.json',{'plan_sha256':digest(output/'plan.json'),'items':records,
             'now_consumed_regression':True,'no_fitting_selection_user_audio_or_scores':True})
    guardian = BassHarmonic().models[1]
    p = [probabilities(network,i['frames'],i['attack_frames'],i['release_frames'],i['x'],normalizer) for i in items]
    g = [guardian.probability(i['base_x']) for i in items]
    raw = evaluate(items,p,g,winner['threshold'],winner['guardian_threshold'])
    guarded = evaluate(items,p,g,margin(winner['threshold']),winner['guardian_threshold'])
    result = {'passes':bool(raw['passes'] and guarded['passes']
                           and raw['false_notes_removed']>0
                           and guarded['false_notes_removed']>0),
              'raw':raw,'margin':guarded,'first_pass_complete':True,'now_consumed_regression':True,
              'no_retuning':True,'recordings':len(items),'checkpoint_sha256':winner['checkpoint_sha256'],
              'batch_selection_sha256':digest(run/'batch-selection.json'),
              'test_manifest_sha256':digest(output/'manifest.json'),'scope':fixture_plan['scope'],
              'native_baseline_metadata':engine.bass_verification}
    preserve(report_path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','margin','native_baseline_metadata')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    verify(args.run,args.output)
