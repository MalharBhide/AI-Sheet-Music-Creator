"""One first-pass original MP3 check of the frozen acoustic-temporal candidate winner."""

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
from bass_temporal_v16_release import load_frozen
from bass_training_data import eligible
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from train_bass_consensus import score
from train_bass_temporal_v16 import FRESH_SEEDS, predictions

SEEDS = FRESH_SEEDS


def require_regression(run, winner):
    report = json.loads((run / 'consumed-regression.json').read_text())
    if (not report['passes'] or report['false_notes_removed'] <= 0 or len(report['per_recording']) != 252
            or any(not row['passes'] for row in report['per_recording'])
            or report['checkpoint_sha256'] != winner['checkpoint_sha256']
            or report['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or (report['threshold'], report['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])):
        raise ValueError('Failed or changed consumed temporal regression')
    piano = json.loads((run / 'piano-regression.json').read_text())
    if (not piano['passes'] or any(not r['passes'] for r in piano['per_recording'])
            or not piano['all_declared_sources_checked'] or not piano['no_retuning_or_replacement']
            or not piano['now_consumed_regression'] or piano['checkpoint_sha256'] != winner['checkpoint_sha256']
            or piano['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or (piano['threshold'], piano['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])):
        raise ValueError('Failed or changed reserved real-piano regression')
    folder = run.parent / 'v16-piano-regression-v1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    declared = {'v16-piano-regression-' + source['id'] + '-' + variant
                for source in plan['sources'] for variant in plan['variants']}
    recorded = [r['id'] for r in manifest['items']]
    excluded = [r['id'] for r in manifest['excluded_no_bass']]
    reported = [r['id'] for r in piano['per_recording']]
    if (piano['manifest_sha256'] != digest(folder / 'manifest.json')
            or manifest['plan_sha256'] != digest(folder / 'plan.json')
            or plan['checkpoint_sha256'] != winner['checkpoint_sha256']
            or plan['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or plan['prior_regression_sha256'] != digest(run / 'consumed-regression.json')
            or plan['producer_sha256'] != digest(Path(__file__).with_name('evaluate_v16_piano_regression.py'))
            or not manifest['now_consumed_regression'] or not plan['no_fitting_selection_or_user_audio']
            or plan['variants'] != ['piano-mp3', 'demucs-bass'] or not recorded
            or len(recorded + excluded) != len(declared) or set(recorded + excluded) != declared
            or len(reported) != len(recorded) or set(reported) != set(recorded)
            or any(not r['excluded_no_bass'] or not r['id'].endswith('-demucs-bass')
                   or r['plan_sha256'] != manifest['plan_sha256'] for r in manifest['excluded_no_bass'])):
        raise ValueError('Incomplete or changed reserved real-piano sources')
    return digest(run / 'consumed-regression.json')


def verify(run, output):
    report_path = run / 'original-first-pass.json'
    if report_path.exists():
        raise ValueError('Preserve first-pass temporal result; no retuning or replacement test')
    plan, winner, model, normalizer = load_frozen(run)
    regression = require_regression(run, winner)
    cache = json.loads((Path(plan['baseline_root']) / 'manifest.json').read_text())
    consumed_groups = {row['source_group'] for row in cache['items']}
    if consumed_groups & {'original-bass-seed-' + str(seed) for seed in SEEDS}:
        raise ValueError('First-pass seeds overlap prior fitting, selection or consumed tests')
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_names = ('training/fresh_bass_temporal_v16.py', 'training/prepare_bass_positive_data.py',
                    'training/prepare_slakh_bass.py', 'training/bass_temporal_v15_data.py', 'backend/app/services/temporal_note_model.py',
                    'training/prepare_bass_texture_v12.py', 'training/harmonic_candidate_features.py',
                    'backend/app/services/piano_transcription.py', 'backend/app/services/note_evidence.py')
    fresh_plan = {'seeds': list(SEEDS), 'checkpoint_sha256': winner['checkpoint_sha256'],
                  'batch_selection_sha256': digest(run / 'batch-selection.json'),
                  'consumed_regression_sha256': regression,
                  'piano_regression_sha256': digest(run / 'piano-regression.json'), 'baseline_hashes': plan['baseline_hashes'],
                  'code_sha256': {name: digest(root / name) for name in source_names},
                  'rights': 'Project-authored procedural signals and note sequences',
                  'generator_partitions': {'original': list(SEEDS[:16]), 'texture': list(SEEDS[16:])},
                  'positive_first_pass_gain_required': True,
                  'mp3_bitrate_kbps_cycle': [48, 128, 192], 'rate': RATE,
                  'no_fitting_selection_user_audio_or_scores': True,
                  'scope': 'Unseen seeds of the two existing generator distributions, not independent human-song evidence. Native V15 baseline; one frozen candidate check. All sources become consumed after this first pass.'}
    preserve(output / 'plan.json', fresh_plan)
    engine, evidence, records, items = _GeneralEngine('balanced'), {}, [], []
    native_predict = engine.predict_function

    def capture(*args, **kwargs):
        result = native_predict(*args, **kwargs)
        evidence['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    for index, seed in enumerate(SEEDS):
        name = 'bass-temporal-v16-fresh-v1-' + str(seed)
        work = output / name
        work.mkdir()
        samples, labels = (original if index < 16 else texture)(seed)
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
        frames = sequences(spectrum(samples, rate), events)
        record = {'id': name, 'group': 'test', 'source_group': 'original-bass-seed-' + str(seed),
                  'corpus': 'original-bass-temporal-v16-fresh', 'audio': str(audio),
                  'audio_sha256': digest(audio), 'reference': labels.tolist(), 'pitch_reference': labels.tolist(),
                  'seconds': 29.5, 'duration': 30., 'codec_clock': clock}
        path = work / 'features.npz'
        np.savez_compressed(path, frames=frames, x=x, base_x=base_x, events=events, eligible=eligible(events, 30.),
                            plan_sha256=digest(output / 'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
        items.append({**record, 'x': x, 'frames': frames, 'base_x': base_x, 'events': events, 'eligible': eligible(events, 30.),
                      'reference': labels.copy(), 'pitch_reference': labels.copy()})
        print(json.dumps({'cached': len(records), 'id': name, 'v15_retained': len(events)}), flush=True)
    if fresh_plan['code_sha256'] != {name: digest(root / name) for name in source_names}:
        raise ValueError('First-pass code changed during inference')
    load_frozen(run)
    require_regression(run, winner)
    preserve(output / 'manifest.json', {'plan_sha256': digest(output / 'plan.json'), 'items': records,
             'no_fitting_selection_user_audio_or_scores': True, 'now_consumed_regression': True})
    guardian = BassHarmonic().models[1]
    result = score(items, predictions(model, normalizer, items), [guardian.probability(i['base_x']) for i in items],
                   winner['threshold'], winner['guardian_threshold'])
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
