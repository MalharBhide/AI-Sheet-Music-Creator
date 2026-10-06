"""One MP3 first-pass timing check; no test-driven tuning or user audio."""

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
from attack_local_model import probabilities
from bass_joint_timing_v18 import margin
from bass_training_data import eligible
from evaluate_early_consensus_v22 import frozen
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from score_early_consensus import evaluate
from train_early_consensus_v22 import FRESH_SEEDS


def require_regression(run, winner):
    report = json.loads((run/'consumed-regression.json').read_text())
    if (not report['passes'] or report['recordings']!=328 or not report['consumed_regression']
            or report['checkpoint_sha256']!=winner['checkpoint_sha256']
            or report['batch_selection_sha256']!=digest(run/'batch-selection.json')
            or report['test_used_for_selection'] or not report['no_user_audio_or_scores']):
        raise ValueError('Missing safe complete consumed timing regression')
    for name,t,r,s in (('raw',winner['timing_threshold'],winner['remove_threshold'],1.),
                       ('margin',margin(winner['timing_threshold']),margin(winner['remove_threshold']),.5)):
        evidence = report[name]
        if (not evidence['passes'] or len(evidence['per_recording'])!=328
                or any(not row['passes'] for row in evidence['per_recording'])
                or evidence['onset_error_reduction_seconds']<=1e-6
                or (evidence['timing_threshold'],evidence['remove_threshold'],evidence['strength'])!=(t,r,s)):
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
    previous=json.loads((Path(plan['previous_fresh_root'])/'manifest.json').read_text())
    if ({r['source_group'] for r in consumed['items']} | {r['source_group'] for r in previous['items']}) & {'original-bass-seed-'+str(s) for s in FRESH_SEEDS}:
        raise ValueError('First-pass fixtures overlap fitting/selection/regression sources')
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    names = ('training/fresh_early_consensus_v22.py','training/prepare_bass_positive_data.py',
             'training/prepare_bass_texture_v12.py','training/prepare_slakh_bass.py',
             'backend/app/services/piano_transcription.py','backend/app/services/note_evidence.py',
             'backend/app/services/temporal_note_model.py','backend/app/services/bass_harmonic.py',
             'training/attack_local_features.py','training/attack_local_model.py',
             'training/pitch_consensus_early.py','training/score_early_consensus.py')
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
        name = 'early-consensus-v22-fresh-'+str(seed)
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
        attack_frames = attack_sequences(observe(samples,rate),events)
        selected = eligible(events,30.)
        record = {'id':name,'group':'test','source_group':'original-bass-seed-'+str(seed),
                  'corpus':'original-early-consensus-v22-first-pass','audio':str(audio),'audio_sha256':digest(audio),
                  'duration':30.,'seconds':29.5,'reference':labels.tolist(),'pitch_reference':labels.tolist(),
                  'codec_clock':clock}
        path = work/'features.npz'
        np.savez_compressed(path,attack_frames=attack_frames,frames=frames,x=x,base_x=base_x,events=events,eligible=selected,
                            plan_sha256=digest(output/'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
        items.append({**record,'attack_frames':attack_frames,'events':events,'x':x,'frames':frames,'base_x':base_x,'eligible':selected,
                      'reference':labels.copy(),'pitch_reference':labels.copy()})
        print(json.dumps({'cached':len(records),'id':name,'native_v16_notes':len(events)}),flush=True)
    if fixture_plan['code_sha256']!={n:digest(root/n) for n in names}:
        raise ValueError('First-pass producer changed during inference')
    frozen(run)
    require_regression(run,winner)
    preserve(output/'manifest.json',{'plan_sha256':digest(output/'plan.json'),'items':records,
             'now_consumed_regression':True,'no_fitting_selection_user_audio_or_scores':True})
    guardian = BassHarmonic().models[1]
    p = [probabilities(network,i['frames'],i['attack_frames'],i['x'],normalizer) for i in items]
    g = [guardian.probability(i['base_x']) for i in items]
    raw = evaluate(items,p,g,winner['timing_threshold'],winner['remove_threshold'])
    guarded = evaluate(items,p,g,margin(winner['timing_threshold']),margin(winner['remove_threshold']),.5)
    result = {'passes':bool(raw['passes'] and guarded['passes']
                           and raw['onset_error_reduction_seconds']>1e-6
                           and guarded['onset_error_reduction_seconds']>1e-6),
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
