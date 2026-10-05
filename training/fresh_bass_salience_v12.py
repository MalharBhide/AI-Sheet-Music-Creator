"""One first-pass original MP3 check of the frozen acoustic-salience winner."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from bass_salience_v12_data import features
from bass_salience_v12_release import load_frozen
from bass_training_data import eligible
from prepare_bass_positive_data import RATE, original
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from train_bass_salience_v12 import predictions, score

SEEDS = tuple(range(262301, 262333))


def require_regression(run, winner):
    report = json.loads((run / 'consumed-regression.json').read_text())
    if (not report['passes'] or report['false_notes_removed'] <= 0 or len(report['per_recording']) != 124
            or any(not row['passes'] for row in report['per_recording'])
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or (report['threshold'], report['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])):
        raise ValueError('Failed or changed consumed salience regression')
    return digest(run / 'consumed-regression.json')


def verify(run, output):
    report_path = run / 'original-first-pass.json'
    if report_path.exists():
        raise ValueError('Preserve first-pass salience result; no retuning or replacement test')
    plan, winner, models = load_frozen(run)
    regression = require_regression(run, winner)
    cache = json.loads((Path(plan['baseline_cache_root']) / 'manifest.json').read_text())
    consumed_groups = {row['source_group'] for row in cache['items']}
    if consumed_groups & {'original-bass-seed-' + str(seed) for seed in SEEDS}:
        raise ValueError('First-pass seeds overlap prior fitting, selection or consumed tests')
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_names = ('training/fresh_bass_salience_v12.py', 'training/prepare_bass_positive_data.py',
                    'training/prepare_slakh_bass.py', 'training/bass_salience_v12_data.py',
                    'backend/app/services/piano_transcription.py', 'backend/app/services/note_evidence.py')
    fresh_plan = {'seeds': list(SEEDS), 'checkpoint_sha256': winner['checkpoint_sha256'],
                  'batch_selection_sha256': digest(run / 'batch-selection.json'),
                  'consumed_regression_sha256': regression, 'baseline_hashes': plan['baseline_hashes'],
                  'code_sha256': {name: digest(root / name) for name in source_names},
                  'rights': 'Project-authored procedural signals and note sequences',
                  'mp3_bitrate_kbps_cycle': [48, 128, 192], 'rate': RATE,
                  'no_fitting_selection_user_audio_or_scores': True,
                  'scope': 'Unseen seeds of the existing generator, not independent human-song evidence. Native V12 baseline; one frozen candidate check. All sources become consumed after this first pass.'}
    preserve(output / 'plan.json', fresh_plan)
    engine, evidence, records, items = _GeneralEngine('balanced'), {}, [], []
    native_predict = engine.predict_function

    def capture(*args, **kwargs):
        result = native_predict(*args, **kwargs)
        evidence['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    for index, seed in enumerate(SEEDS):
        name = 'bass-salience-fresh-v1-' + str(seed)
        work = output / name
        work.mkdir()
        samples, labels = original(seed)
        _, clock = encode(samples, RATE, work, fresh_plan['mp3_bitrate_kbps_cycle'][index % 3])
        audio = work / 'decoded.wav'
        samples, rate = sf.read(audio, dtype='float32')
        if rate != RATE or samples.ndim != 1 or len(samples) != 30 * RATE:
            raise ValueError('Changed native first-pass audio clock')
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            midi = engine.predict(audio, 'bass', 90.)
            notes = [note for part in midi.instruments for note in part.notes]
            base_x = note_features(samples, rate, evidence['acoustic'], notes, include_context=True)
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
        x = features(events, base_x)
        record = {'id': name, 'group': 'test', 'source_group': 'original-bass-seed-' + str(seed),
                  'corpus': 'original-bass-salience-fresh', 'audio': str(audio),
                  'audio_sha256': digest(audio), 'reference': labels.tolist(), 'pitch_reference': labels.tolist(),
                  'seconds': 29.5, 'duration': 30., 'codec_clock': clock}
        path = work / 'features.npz'
        np.savez_compressed(path, x=x, base_x=base_x, events=events, eligible=eligible(events, 30.),
                            plan_sha256=digest(output / 'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
        items.append({**record, 'x': x, 'events': events, 'eligible': eligible(events, 30.),
                      'reference': labels.copy(), 'pitch_reference': labels.copy()})
        print(json.dumps({'cached': len(records), 'id': name, 'v12_retained': len(events)}), flush=True)
    if fresh_plan['code_sha256'] != {name: digest(root / name) for name in source_names}:
        raise ValueError('First-pass code changed during inference')
    load_frozen(run)
    require_regression(run, winner)
    preserve(output / 'manifest.json', {'plan_sha256': digest(output / 'plan.json'), 'items': records,
             'no_fitting_selection_user_audio_or_scores': True, 'now_consumed_regression': True})
    result = score(items, *predictions(models, items), winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  test_manifest_sha256=digest(output / 'manifest.json'), first_pass_complete=True,
                  now_consumed_regression=True, no_retuning=True, scope=fresh_plan['scope'],
                  native_baseline_metadata=engine.bass_verification)
    preserve(report_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.run, args.output)
