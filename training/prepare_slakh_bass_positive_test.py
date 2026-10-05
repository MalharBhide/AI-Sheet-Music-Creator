"""Prepare all reserved Slakh bass groups only after a frozen stress-safe winner."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import yaml
from app.services.source_separation import StemSeparator
from bass_positive_release import load_frozen, require_report
from bass_training_data import VERSION, cache_one
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import bass_sources, encode
from slakh_references import crop, intervals


def prepare(directory, run):
    winner, _ = load_frozen(directory, run)
    require_report(run, winner, 'held-regression')
    prior = json.loads((directory / 'plan.json').read_text())
    root = directory.parent / 'babyslakh/data/babyslakh_16k'
    groups = sorted(source for source, group in prior['project_partition'].items() if group == 'test')
    if len(groups) != 4:
        raise ValueError('Reserved evaluation requires all four frozen source groups')
    output = directory.parent / 'slakh-bass-positive-reserved-v2'
    output.mkdir(exist_ok=True)
    plan = {'groups': groups, 'record': 4603870, 'license': 'cc-by-4.0',
            'oracle_offsets': [0, 30, 60, 90], 'demucs_offsets': [0],
            'mp3_bitrate_kbps_cycle': [48, 128, 192], 'candidate_decoder': VERSION,
            'checkpoint_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'training_plan_sha256': digest(directory / 'plan.json'),
            'code_sha256': digest(Path(__file__)),
            'source_hashes': {group: {str(path.relative_to(root / group)): digest(path)
                             for path in sorted((root / group).rglob('*')) if path.is_file()} for group in groups},
            'scope': 'First one-shot bass evaluation of four reserved Slakh groups; no test retuning; not guaranteed composition/pretraining independence.',
            'no_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    separator, items, hashes = None, [], {}
    for group in groups:
        track = root / group
        meta = yaml.safe_load((track / 'metadata.yaml').read_text())
        info = sf.info(track / 'mix.wav')
        if info.samplerate != 16000 or info.channels != 1 or info.duration < 120:
            raise ValueError('Unexpected reserved source clock')
        keys, support, mixed_keys, mixed_support, selected = [], [], [], [], []
        for name, stem in sorted(meta['stems'].items()):
            if stem['is_drum']:
                continue
            audio, midi = track / 'stems' / (name + '.wav'), track / 'MIDI' / (name + '.mid')
            if not audio.is_file() or not midi.is_file():
                continue
            actual = sf.info(audio)
            if actual.samplerate != info.samplerate or actual.frames != info.frames or actual.channels != 1:
                raise ValueError('Reserved paired-source clocks differ')
            k, p = intervals(pretty_midi.PrettyMIDI(str(midi)), info.duration)
            k, p = [values[(values[:, 2] >= 21) & (values[:, 2] < 60)] for values in (k, p)]
            mixed_keys.extend(k.tolist())
            mixed_support.extend(p.tolist())
            if name in bass_sources(meta):
                selected.append(name)
                keys.extend(k.tolist())
                support.extend(p.tolist())
        keys, support, mixed_keys, mixed_support = [np.asarray(sorted(set(map(tuple, rows))), dtype=float).reshape(-1, 3)
                                                   for rows in (keys, support, mixed_keys, mixed_support)]
        for variant, offsets in [('oracle', plan['oracle_offsets']), ('demucs', plan['demucs_offsets'])]:
            if variant == 'oracle' and not selected:
                continue
            for index, offset in enumerate(offsets):
                name = f'slakh-bass-positive-reserved-v2-{group}-{variant}-{offset:03}'
                work = output / name
                work.mkdir(exist_ok=True)
                completed = work / 'complete.json'
                if completed.exists():
                    record = json.loads(completed.read_text())
                    item = record['item']
                    if (record['plan_sha256'] != digest(output / 'plan.json')
                            or record['audio_sha256'] != digest(Path(item['audio']))
                            or record['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                        raise ValueError('Changed completed reserved evidence')
                else:
                    samples = np.zeros(30 * info.samplerate, dtype=np.float32)
                    paths = [track / 'mix.wav'] if variant == 'demucs' else [track / 'stems' / (source + '.wav') for source in selected]
                    for audio in paths:
                        values, rate = sf.read(audio, start=offset * info.samplerate, frames=len(samples), dtype='float32')
                        if rate != info.samplerate or len(values) != len(samples) or not np.isfinite(values).all():
                            raise ValueError('Reserved crop clock differs')
                        samples += values
                    if np.std(samples) < 1e-6:
                        raise ValueError('Silent reserved crop cannot calibrate MP3 clock')
                    bitrate = plan['mp3_bitrate_kbps_cycle'][index % 3]
                    decoded, clock = encode(samples, info.samplerate, work, bitrate)
                    audio = work / 'decoded.wav'
                    if variant == 'demucs':
                        if separator is None:
                            separator = StemSeparator()
                        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                            separated = separator.separate(audio, work)
                        if 'bass' not in separated:
                            sf.write(work / 'bass.wav', np.zeros_like(decoded), 22050, subtype='FLOAT')
                        audio = work / 'bass.wav'
                    duration = min(30., sf.info(audio).duration)
                    if abs(duration - len(decoded) / 22050) > 1 / 22050:
                        raise ValueError('Reserved separator clock changed')
                    reference, pitch = crop(mixed_keys if variant == 'demucs' else keys,
                                            mixed_support if variant == 'demucs' else support, offset, duration)
                    item = {'id': name, 'source_group': group, 'group': 'test', 'corpus': 'slakh-' + variant + '-bass',
                            'audio': str(audio.resolve()), 'reference': reference.tolist(), 'pitch_reference': pitch.tolist(),
                            'candidate_decoder': VERSION, 'evaluation_window': [.25, duration - .25]}
                    cache = cache_one(output, item)
                    record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                              'audio_sha256': digest(audio), 'cache_sha256': digest(cache),
                              'codec_clock': clock, 'bitrate_kbps': bitrate}
                    preserve(completed, record)
                items.append(item)
                hashes[name] = record['cache_sha256']
                print(json.dumps({'cached': len(items), 'id': name}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': hashes,
             'plan_sha256': digest(output / 'plan.json'), 'scope': plan['scope'], 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.run)
