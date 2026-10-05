"""Add real piano/MP3 counterexamples while preserving corrected clocks/splits."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.source_separation import StemSeparator
from app.services.temporal_note_model import sequences, spectrum
from bass_training_data import eligible, identity
from current_bass_v15_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from prepare_vienna import CHECKSUMS
from slakh_references import crop, intervals

OFFSET = 30
DEMUCS_PLAYERS = (1, 14, 15, 18)


def declared_sources(directory):
    provenance = json.loads((directory / 'vienna-source.json').read_text())
    if (provenance['author'] != 'Werner Goebl' or provenance['license'] != 'CC-BY-4.0'
            or {r['url']: r['sha256'] for r in provenance['archives']} != CHECKSUMS):
        raise ValueError('Changed verified Vienna source rights/checksums')
    # The root vienna-clock-audit.json is historical and has incorrect trimmed
    # Chopin anchors. Use only the corrected V3 calibration, never predictions.
    clocks = json.loads((directory / 'note-verifier-v3-data/clock-audit.json').read_text())
    if len(clocks) != 88 or len({r['id'] for r in clocks}) != 88:
        raise ValueError('Incomplete corrected Vienna clock audit')
    records = []
    for clock in clocks:
        name = clock['id'].removeprefix('vienna-')
        player = int(name.rsplit('_p', 1)[1])
        group = 'train' if player <= 14 else 'validation' if player <= 18 else 'test'
        if clock['group'] != group or not 1 <= player <= 22 or '1st-3rd' in name:
            raise ValueError('Changed Vienna performer split')
        if group == 'test':
            continue
        sources = list((directory / 'vienna/audio').rglob(name + '.wav'))
        if len(sources) != 1:
            raise ValueError('Missing or ambiguous Vienna audio pair')
        audio, midi = sources[0], directory / 'vienna/midi' / (name + '.mid')
        if sf.info(audio).duration < OFFSET + 30:
            continue
        records.append({'id': name, 'group': group, 'player': player,
                        'source_group': 'vienna-performance-' + name,
                        'audio': str(audio), 'midi': str(midi),
                        'audio_sha256': digest(audio), 'midi_sha256': digest(midi),
                        'shift_seconds': clock['initial_shift'] + clock['spectral_correction'],
                        'variants': ['piano-mp3', 'demucs-bass'] if player in DEMUCS_PLAYERS else ['piano-mp3']})
    if {g: sum(r['group'] == g for r in records) for g in ('train', 'validation')} != {'train': 42, 'validation': 12}:
        raise ValueError('Changed predeclared eligible Vienna cohort')
    return records


def reference(record):
    midi = pretty_midi.PrettyMIDI(record['midi'])
    duration = max(midi.get_end_time(), sf.info(record['audio']).duration - record['shift_seconds'])
    keys, support = intervals(midi, duration)
    for rows in (keys, support):
        rows[:, :2] += record['shift_seconds']
    keys = keys[(keys[:, 2] >= 21) & (keys[:, 2] < 60)]
    support = support[(support[:, 2] >= 21) & (support[:, 2] < 60)]
    return crop(keys, support, OFFSET, 30.)


def prepare(directory, output):
    root = Path(__file__).resolve().parents[1]
    declared = declared_sources(directory)
    names = ('training/prepare_vienna_bass_v15.py', 'training/prepare_slakh_bass.py',
             'training/slakh_references.py', 'backend/app/services/source_separation.py',
             'backend/app/services/note_evidence.py', 'backend/app/services/temporal_note_model.py')
    plan = {'baseline_hashes': hashes(), 'sources': declared, 'offset': OFFSET, 'duration': 30.,
            'corrected_clock_audit_sha256': digest(directory / 'note-verifier-v3-data/clock-audit.json'),
            'source_provenance_sha256': digest(directory / 'vienna-source.json'),
            'license': 'CC-BY-4.0', 'author': 'Werner Goebl', 'doi': '10.21939/4X22',
            'demucs_players': list(DEMUCS_PLAYERS), 'mp3_bitrate_kbps_cycle': [48, 128, 192],
            'code_sha256': {name: digest(root / name) for name in names},
            'no_test_audio_or_metrics': True, 'no_user_audio_or_scores': True,
            'labels': 'Instrument key releases plus CC64 sustained pitch support; overlapping pre-crop holds protected. Corrected 0–15s clock calibration excluded from the 30–60s crops.',
            'scope': 'All eligible existing train/validation performers; original piano-register MP3 counterexamples plus predeclared Demucs bass variants matching Full-song input. Shared compositions/instrument and reused development performers; not independent test accuracy.',
            'silent_demucs_policy': 'If production separation emits no bass, audit exclusion; never fit a silent fallback as musical negatives.'}
    output.mkdir(exist_ok=True)
    preserve(output / 'plan.json', plan)
    engine, separator, evidence = _GeneralEngine('balanced'), None, {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        evidence['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    records, audit = [], []
    for index, source in enumerate(declared):
        if digest(Path(source['audio'])) != source['audio_sha256'] or digest(Path(source['midi'])) != source['midi_sha256']:
            raise ValueError('Changed declared Vienna source')
        keys, pitch = reference(source)
        for variant in source['variants']:
            name = 'vienna-bass-v15-' + source['id'] + '-' + variant
            work = output / name
            work.mkdir(exist_ok=True)
            done = work / 'complete.json'
            if done.exists():
                completed = json.loads(done.read_text())
                if completed['plan_sha256'] != digest(output / 'plan.json'):
                    raise ValueError('Changed completed Vienna preparation')
                if completed.get('excluded_no_bass'):
                    audit.append(completed)
                    continue
                row = completed['item']
                if digest(Path(row['audio'])) != row['audio_sha256'] or digest(work / 'features.npz') != row['cache_sha256']:
                    raise ValueError('Changed completed Vienna bass evidence')
                records.append(row)
                continue
            samples, rate = librosa.load(source['audio'], sr=22050, offset=OFFSET, duration=30)
            if rate != 22050 or len(samples) != 30*rate or not np.isfinite(samples).all():
                raise ValueError('Changed real piano crop clock')
            _, clock = encode(samples, rate, work, plan['mp3_bitrate_kbps_cycle'][index % 3])
            audio = work / 'decoded.wav'
            if variant == 'demucs-bass':
                separator = separator or StemSeparator()
                with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                    stems = separator.separate(audio, work)
                if 'bass' not in stems:
                    skipped = {'id': name, 'group': source['group'], 'excluded_no_bass': True,
                               'plan_sha256': digest(output / 'plan.json')}
                    preserve(done, skipped)
                    audit.append(skipped)
                    continue
                audio = stems['bass']
            waveform, rate = sf.read(audio, dtype='float32')
            if rate != 22050 or waveform.ndim != 1 or len(waveform) != 30*rate:
                raise ValueError('Separation shifted the real piano clock')
            with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                midi = engine.predict(audio, 'bass', 90.)
                notes = [n for part in midi.instruments for n in part.notes]
                base_x = note_features(waveform, rate, evidence['acoustic'], notes, include_context=True)
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
            frames = sequences(spectrum(waveform, rate), events)
            row = {'id': name, 'source_group': source['source_group'], 'group': source['group'],
                   'corpus': 'vienna-' + variant + '-bass', 'audio': str(audio),
                   'audio_sha256': digest(audio), 'reference': keys.tolist(), 'pitch_reference': pitch.tolist(),
                   'duration': 30., 'seconds': 29.5, 'player': source['player'], 'variant': variant,
                   'codec_clock': clock, 'retained_v15': len(events)}
            np.savez_compressed(work / 'features.npz', events=events, base_x=base_x, frames=frames,
                                eligible=eligible(events, 30.), identity=identity(row),
                                plan_sha256=digest(output / 'plan.json'))
            row['cache_sha256'] = digest(work / 'features.npz')
            preserve(done, {'plan_sha256': digest(output / 'plan.json'), 'item': row})
            records.append(row)
            print(json.dumps({'cached': len(records), 'id': name, 'retained_v15': len(events)}), flush=True)
    if plan['baseline_hashes'] != hashes() or plan['code_sha256'] != {name: digest(root / name) for name in names}:
        raise ValueError('Vienna fitting preparation changed during inference')
    preserve(output / 'manifest.json', {'plan_sha256': digest(output / 'plan.json'), 'items': records,
             'no_test_audio_or_metrics': True, 'no_user_audio_or_scores': True})
    preserve(output / 'source-audit.json', {'excluded': audit, 'baseline_metadata': engine.bass_verification})
    print(json.dumps({'completed': len(records), 'excluded_no_bass': len(audit),
                     'groups': {g: sum(r['group'] == g for r in records) for g in ('train', 'validation')}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.output)
