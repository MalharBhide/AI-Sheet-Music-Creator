"""Prepare CC BY bass supervision using the source partition frozen before V10.

Only training/validation groups are inferred. Oracle labels come from rendered
bass MIDI. Demucs labels protect low notes from every rendered pitched source,
including piano/guitar that can cross into its bass output.
"""

import argparse
import contextlib
import json
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import yaml
from app.services.source_separation import StemSeparator
from bass_training_data import VERSION, cache_one
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_training import ARCHIVE_MD5, verify_codec
from slakh_references import crop, intervals


def bass_sources(metadata):
    return [name for name, stem in sorted(metadata['stems'].items())
            if not stem['is_drum'] and 32 <= stem['program_num'] < 40]


def encode(samples, rate, work, bitrate):
    samples = librosa.resample(samples, orig_sr=rate, target_sr=22050)
    samples *= .95 / max(.95, float(np.max(np.abs(samples))))
    sf.write(work / 'original.wav', samples, 22050, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'original.wav'),
                    '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k', str(work / 'source.mp3')], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'source.mp3'),
                    '-ar', '22050', '-ac', '1', '-c:a', 'pcm_f32le', str(work / 'decoded.wav')], check=True)
    decoded, _ = sf.read(work / 'decoded.wav', dtype='float32')
    return decoded, verify_codec(samples, decoded, 22050)


def prepare(directory):
    baseline_plan = directory / 'slakh-training-v1/plan.json'
    previous = json.loads(baseline_plan.read_text())
    root = directory / 'babyslakh'
    record = json.loads((root / 'record.json').read_text())
    checksum = json.loads((root / 'checksum.json').read_text())
    if (record['id'] != 4603870 or record['metadata']['license']['id'] != 'cc-by-4.0'
            or checksum['md5'] != ARCHIVE_MD5 or checksum['bytes'] != 882818115):
        raise ValueError('Official dataset rights/checksum changed')
    sources = root / 'data/babyslakh_16k'
    groups = previous['project_partition']
    output = directory / 'slakh-bass-v1'
    output.mkdir(exist_ok=True)
    plan = {'partition_plan_sha256': digest(baseline_plan), 'source_hashes': previous['source_hashes'],
            'project_partition': groups, 'record': 4603870, 'license': 'cc-by-4.0',
            'oracle_offsets': [0, 30, 60, 90], 'demucs_offsets': [0],
            'mp3_bitrate_kbps_cycle': [48, 128, 192], 'candidate_decoder': VERSION,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                            ('prepare_slakh_bass.py', 'bass_training_data.py', 'slakh_references.py')},
            'separator_sha256': digest(Path(__file__).resolve().parents[1] / 'backend/app/services/source_separation.py'),
            'scope': 'Fixed synthesized source groups. Exact source MIDI for oracle and full-mixture Demucs bass; no user audio.',
            'labels': 'Oracle: rendered bass sources. Demucs: all rendered pitched sources in 21-59, protecting cross-stem piano/guitar. Key releases and CC64/crop support separate.',
            'missing_bass_source': 'Skip oracle variant when no actual rendered bass pair exists; still prepare full-mixture Demucs with all actual low-pitch labels.',
            'no_test_inference': True, 'no_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    separator, items, cache_hashes, audit = None, [], {}, []
    for identity in sorted(groups):
        if groups[identity] == 'test':
            continue
        track = sources / identity
        for relative, expected in plan['source_hashes'][identity].items():
            if digest(track / relative) != expected:
                raise ValueError('Changed frozen rendered-source data')
        meta = yaml.safe_load((track / 'metadata.yaml').read_text())
        info = sf.info(track / 'mix.wav')
        if info.samplerate != 16000 or info.channels != 1 or info.duration < 120:
            raise ValueError('Unexpected source mix clock')
        keys, support, mixed_keys, mixed_support = [], [], [], []
        selected = []
        for name, stem in sorted(meta['stems'].items()):
            if stem['is_drum']:
                continue
            audio, midi = track / 'stems' / (name + '.wav'), track / 'MIDI' / (name + '.mid')
            if not audio.is_file() or not midi.is_file():
                audit.append({'track': identity, 'source': name, 'actual_source_files_present': False,
                              'excluded_unrendered_source': True})
                continue
            actual = sf.info(audio)
            if actual.samplerate != info.samplerate or actual.frames != info.frames or actual.channels != 1:
                raise ValueError('Bass source audio clock differs from mix')
            k, p = intervals(pretty_midi.PrettyMIDI(str(midi)), info.duration)
            k = k[(k[:, 2] >= 21) & (k[:, 2] < 60)]
            p = p[(p[:, 2] >= 21) & (p[:, 2] < 60)]
            mixed_keys.extend(k.tolist())
            mixed_support.extend(p.tolist())
            if name in bass_sources(meta):
                selected.append(name)
                keys.extend(k.tolist())
                support.extend(p.tolist())
            audit.append({'track': identity, 'source': name, 'class': meta['stems'][name]['inst_class'],
                          'audio_sha256': digest(audio), 'midi_sha256': digest(midi), 'notes': len(k),
                          'metadata_audio_rendered': meta['stems'][name]['audio_rendered'],
                          'actual_source_files_present': True})
        keys = np.asarray(sorted(set(map(tuple, keys))), dtype=float).reshape(-1, 3)
        support = np.asarray(sorted(set(map(tuple, support))), dtype=float).reshape(-1, 3)
        mixed_keys = np.asarray(sorted(set(map(tuple, mixed_keys))), dtype=float).reshape(-1, 3)
        mixed_support = np.asarray(sorted(set(map(tuple, mixed_support))), dtype=float).reshape(-1, 3)
        for variant, offsets in [('oracle', plan['oracle_offsets']), ('demucs', plan['demucs_offsets'])]:
            if variant == 'oracle' and not selected:
                continue
            for index, offset in enumerate(offsets):
                name = f'slakh-bass-v1-{identity}-{variant}-{offset:03}'
                work = output / name
                work.mkdir(exist_ok=True)
                completion = work / 'complete.json'
                if completion.exists():
                    completed = json.loads(completion.read_text())
                    item = completed['item']
                    if (completed['plan_sha256'] != digest(output / 'plan.json')
                            or completed['audio_sha256'] != digest(Path(item['audio']))
                            or completed['cache_sha256'] != digest(output / 'features' / (name + '.npz'))):
                        raise ValueError('Changed completed bass preparation')
                else:
                    samples = np.zeros(30 * info.samplerate, dtype=np.float32)
                    audio_paths = [track / 'mix.wav'] if variant == 'demucs' else [track / 'stems' / (s + '.wav') for s in selected]
                    for audio in audio_paths:
                        values, rate = sf.read(audio, start=offset * info.samplerate, frames=len(samples), dtype='float32')
                        if len(values) != len(samples) or rate != info.samplerate or not np.isfinite(values).all():
                            raise ValueError('Bass crop clock mismatch')
                        samples += values
                    if np.std(samples) < 1e-6:
                        raise ValueError('Silent source crop requires an explicit revised plan, not unverified codec timing')
                    bitrate = plan['mp3_bitrate_kbps_cycle'][index % 3]
                    decoded, clock = encode(samples, info.samplerate, work, bitrate)
                    audio = work / 'decoded.wav'
                    if variant == 'demucs':
                        if separator is None:
                            separator = StemSeparator()
                        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                            paths = separator.separate(audio, work)
                        if 'bass' not in paths:
                            sf.write(work / 'bass.wav', np.zeros_like(decoded), 22050, subtype='FLOAT')
                        audio = work / 'bass.wav'
                    duration = min(30., sf.info(audio).duration)
                    if abs(duration - len(decoded) / 22050) > 1 / 22050:
                        raise ValueError('Separator shifted the source clock')
                    reference, pitch = crop(mixed_keys if variant == 'demucs' else keys,
                                            mixed_support if variant == 'demucs' else support, offset, duration)
                    item = {'id': name, 'group': groups[identity], 'source_group': identity,
                            'corpus': 'slakh-' + variant + '-bass', 'audio': str(audio.resolve()),
                            'reference': reference.tolist(), 'pitch_reference': pitch.tolist(),
                            'evaluation_window': [.25, duration - .25], 'candidate_decoder': VERSION}
                    path = cache_one(output, item)
                    completed = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                                 'audio_sha256': digest(audio), 'cache_sha256': digest(path),
                                 'codec_clock': clock, 'bitrate_kbps': bitrate}
                    preserve(completion, completed)
                items.append(item)
                cache_hashes[name] = completed['cache_sha256']
                print(json.dumps({'cached': len(items), 'id': name, 'group': groups[identity]}), flush=True)
    preserve(output / 'source-audit.json', {'items': audit})
    preserve(output / 'manifest.json', {'items': items, 'cache_sha256': cache_hashes,
             'plan_sha256': digest(output / 'plan.json'), 'scope': plan['scope'],
             'no_test_inference': True, 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
