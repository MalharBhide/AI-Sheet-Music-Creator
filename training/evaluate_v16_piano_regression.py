"""One frozen-candidate regression on reserved real piano performers; no fitting."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from app.services.bass_harmonic import BassHarmonic, features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.source_separation import StemSeparator
from app.services.temporal_note_model import sequences, spectrum
from bass_temporal_v16_release import load_frozen
from bass_training_data import eligible
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from prepare_vienna_bass_v15 import reference
from train_bass_consensus import score
from train_bass_temporal_v16 import predictions


def old_regression(run, winner):
    result = json.loads((run / 'consumed-regression.json').read_text())
    if (not result['passes'] or result['false_notes_removed'] <= 0 or len(result['per_recording']) != 252
            or any(not row['passes'] for row in result['per_recording'])
            or result['checkpoint_sha256'] != winner['checkpoint_sha256']
            or result['batch_selection_sha256'] != digest(run / 'batch-selection.json')
            or (result['threshold'], result['guardian_threshold']) != (winner['threshold'], winner['guardian_threshold'])):
        raise ValueError('Failed or changed frozen consumed regression')
    return digest(run / 'consumed-regression.json')


def prepare(directory, run, output):
    if (run / 'piano-regression.json').exists():
        raise ValueError('Preserve first reserved-piano result; no retuning or replacement check')
    fitting, winner, model, normalizer = load_frozen(run)
    old_sha = old_regression(run, winner)
    piano_plan = json.loads((Path(fitting['piano_root']) / 'plan.json').read_text())
    clocks_path = directory / 'note-verifier-v3-data/clock-audit.json'
    if (digest(clocks_path) != piano_plan['corrected_clock_audit_sha256']
            or digest(directory / 'vienna-source.json') != piano_plan['source_provenance_sha256']):
        raise ValueError('Changed corrected clocks or verified piano provenance')
    declared = []
    for clock in json.loads(clocks_path.read_text()):
        if clock['group'] != 'test':
            continue
        name = clock['id'].removeprefix('vienna-')
        player = int(name.rsplit('_p', 1)[1])
        if not 19 <= player <= 22 or '1st-3rd' in name:
            raise ValueError('Changed reserved performer partition')
        sources = list((directory / 'vienna/audio').rglob(name + '.wav'))
        if len(sources) != 1:
            raise ValueError('Ambiguous reserved piano source')
        audio, midi = sources[0], directory / 'vienna/midi' / (name + '.mid')
        if sf.info(audio).duration < 60.:
            continue
        declared.append({'id': name, 'group': 'test', 'player': player,
                         'source_group': 'vienna-performance-' + name, 'audio': str(audio), 'midi': str(midi),
                         'audio_sha256': digest(audio), 'midi_sha256': digest(midi),
                         'shift_seconds': clock['initial_shift'] + clock['spectral_correction']})
    if not declared or {r['source_group'] for r in declared} & {r['source_group'] for r in piano_plan['sources']}:
        raise ValueError('Empty or overlapping real piano regression')
    output.mkdir(exist_ok=True)
    binding = {'checkpoint_sha256': winner['checkpoint_sha256'], 'sources': declared,
               'batch_selection_sha256': digest(run / 'batch-selection.json'), 'prior_regression_sha256': old_sha,
               'corrected_clock_audit_sha256': digest(clocks_path), 'producer_sha256': digest(Path(__file__)),
               'variants': ['piano-mp3', 'demucs-bass'], 'offset': 30, 'duration': 30.,
               'mp3_bitrate_kbps_cycle': [48, 128, 192], 'no_fitting_selection_or_user_audio': True,
               'scope': 'All eligible existing reserved performers, original MP3 and production Demucs bass. These performers are consumed development regressions, not independent human-song accuracy.'}
    preserve(output / 'plan.json', binding)
    engine, separator, evidence = _GeneralEngine('balanced'), None, {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        evidence['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    records, excluded, items = [], [], []
    for index, source in enumerate(declared):
        if digest(Path(source['audio'])) != source['audio_sha256'] or digest(Path(source['midi'])) != source['midi_sha256']:
            raise ValueError('Changed declared reserved piano source')
        keys, pitch = reference(source)
        for variant in binding['variants']:
            name = 'v16-piano-regression-' + source['id'] + '-' + variant
            work = output / name
            work.mkdir(exist_ok=True)
            done = work / 'complete.json'
            if done.exists():
                completed = json.loads(done.read_text())
                if completed['plan_sha256'] != digest(output / 'plan.json'):
                    raise ValueError('Changed reserved piano completion')
                if completed.get('excluded_no_bass'):
                    excluded.append(completed)
                    continue
                row = completed['item']
            else:
                samples, rate = librosa.load(source['audio'], sr=22050, offset=30, duration=30)
                if rate != 22050 or len(samples) != 30*rate:
                    raise ValueError('Changed reserved piano crop clock')
                _, clock = encode(samples, rate, work, binding['mp3_bitrate_kbps_cycle'][index % 3])
                audio = work / 'decoded.wav'
                if variant == 'demucs-bass':
                    separator = separator or StemSeparator()
                    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                        stems = separator.separate(audio, work)
                    if 'bass' not in stems:
                        omitted = {'id': name, 'excluded_no_bass': True, 'plan_sha256': digest(output / 'plan.json')}
                        preserve(done, omitted)
                        excluded.append(omitted)
                        continue
                    audio = stems['bass']
                waveform, rate = sf.read(audio, dtype='float32')
                if rate != 22050 or waveform.ndim != 1 or len(waveform) != 30*rate:
                    raise ValueError('Separation shifted reserved piano clock')
                with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                    midi = engine.predict(audio, 'bass', 90.)
                    notes = [n for part in midi.instruments for n in part.notes]
                    base_x = note_features(waveform, rate, evidence['acoustic'], notes, include_context=True)
                events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
                frames = sequences(spectrum(waveform, rate), events)
                row = {'id': name, 'group': 'test', 'source_group': source['source_group'],
                       'corpus': 'reserved-vienna-' + variant + '-bass', 'audio': str(audio),
                       'audio_sha256': digest(audio), 'reference': keys.tolist(), 'pitch_reference': pitch.tolist(),
                       'duration': 30., 'seconds': 29.5, 'codec_clock': clock}
                np.savez_compressed(work / 'features.npz', events=events, base_x=base_x,
                                    x=features(events, base_x), frames=frames, eligible=eligible(events, 30.),
                                    plan_sha256=digest(output / 'plan.json'))
                row['cache_sha256'] = digest(work / 'features.npz')
                preserve(done, {'item': row, 'plan_sha256': digest(output / 'plan.json')})
                print(json.dumps({'cached': len(records)+1, 'id': name}), flush=True)
            if digest(Path(row['audio'])) != row['audio_sha256'] or digest(work / 'features.npz') != row['cache_sha256']:
                raise ValueError('Changed reserved piano evidence')
            with np.load(work / 'features.npz', allow_pickle=False) as saved:
                if str(saved['plan_sha256']) != digest(output / 'plan.json'):
                    raise ValueError('Changed reserved piano feature identity')
                item = {**row, **{k: saved[k] for k in ('events', 'base_x', 'x', 'frames', 'eligible')}}
            item.update(reference=np.asarray(row['reference'], float).reshape(-1, 3),
                        pitch_reference=np.asarray(row['pitch_reference'], float).reshape(-1, 3))
            items.append(item)
            records.append(row)
    load_frozen(run)
    old_regression(run, winner)
    if digest(Path(__file__)) != binding['producer_sha256']:
        raise ValueError('Piano regression code changed during inference')
    preserve(output / 'manifest.json', {'plan_sha256': digest(output / 'plan.json'), 'items': records,
             'excluded_no_bass': excluded, 'now_consumed_regression': True})
    guardian = BassHarmonic().models[1]
    result = score(items, predictions(model, normalizer, items), [guardian.probability(i['base_x']) for i in items],
                   winner['threshold'], winner['guardian_threshold'])
    result.update(checkpoint_sha256=winner['checkpoint_sha256'],
                  batch_selection_sha256=digest(run / 'batch-selection.json'),
                  manifest_sha256=digest(output / 'manifest.json'), all_declared_sources_checked=True,
                  no_retuning_or_replacement=True, now_consumed_regression=True, scope=binding['scope'])
    preserve(run / 'piano-regression.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_recording'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'run', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.run, args.output)
