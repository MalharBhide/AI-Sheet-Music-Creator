"""Prepare licensed polyphonic accompaniment data with MIDI-derived pitch support.

Uses fixed training/validation sources only, keeping test groups untouched. The
submix is an oracle non-bass/non-drum accompaniment, not a Demucs benchmark.
"""

import argparse
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import yaml
from cache_note_verifier import cache_one
from prepare_robust_training_stems import codec_clock, digest, preserve
from slakh_references import crop, intervals, partition

ARCHIVE_MD5 = '311096dc2bde7d61c97e930edbfc7f78'


def verify_codec(original, decoded, rate):
    """Verify an energetic fixed waveform segment when the intro is silent."""
    if abs(len(original) - len(decoded)) > .03 * rate:
        raise ValueError('MP3 gapless duration check failed')
    starts = range(0, max(1, min(len(original), len(decoded)) - rate + 1), rate)
    anchor = max(starts, key=lambda start: float(np.std(original[start:start + rate])))
    stop = min(len(original), len(decoded), anchor + 5 * rate)
    clock = codec_clock(original[anchor:stop], decoded[anchor:stop], rate)
    clock.update(anchor_sample=anchor, input_frames=len(original), decoded_frames=len(decoded))
    return clock


def prepare(directory):
    root = directory / 'babyslakh'
    metadata = json.loads((root / 'record.json').read_text())
    checksum = json.loads((root / 'checksum.json').read_text())
    if metadata['id'] != 4603870 or metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Official dataset revision/license changed')
    if checksum['md5'] != ARCHIVE_MD5 or checksum['bytes'] != 882818115:
        raise ValueError('Dataset archive checksum not verified')
    sources = root / 'data/babyslakh_16k'
    identities = sorted(path.name for path in sources.glob('Track*'))
    duplicates = json.loads((root / 'duplicates.json').read_text())
    groups = partition(identities, duplicates)
    output = directory / 'slakh-training-v1'
    output.mkdir(exist_ok=True)
    source_hashes = {identity: {str(path.relative_to(sources / identity)): digest(path)
                    for path in sorted((sources / identity).rglob('*')) if path.is_file()}
                     for identity in identities if groups[identity] != 'test'}
    plan = {'record': 4603870, 'archive_md5': ARCHIVE_MD5, 'license': 'cc-by-4.0',
            'source_hashes': source_hashes,
            'source_code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                                   ('prepare_slakh_training.py', 'slakh_references.py', 'cache_note_verifier.py')},
            'duplicate_metadata_sha256': digest(root / 'duplicates.json'),
            'publisher_split_sha256': digest(root / 'redux.json'), 'project_partition': groups,
            'offsets': [0, 30, 60, 90], 'mp3_bitrates': [48, 128, 192],
            'selection': '12/4/4 source duplicate groups, hash-seeded before inference; train/validation only cached',
            'scope': 'Small synthesized prototype, oracle pitched submix; not real-song or production separation accuracy',
            'labels': 'Exact rendered-source MIDI, key releases separate from pedal/crop pitch support',
            'no_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    items, audit = [], []
    for identity in identities:
        if groups[identity] == 'test':
            continue
        track = sources / identity
        meta = yaml.safe_load((track / 'metadata.yaml').read_text())
        if meta['UUID'] != duplicates[identity]['midi_md5']:
            raise ValueError('Publisher duplicate identity differs from source metadata')
        info = sf.info(track / 'mix.wav')
        if info.samplerate != 16000 or info.channels != 1 or info.duration < 120:
            raise ValueError('Unexpected prototype audio clock or format')
        selected, keys, support = [], [], []
        for name, stem in sorted(meta['stems'].items()):
            audio, midi = track / 'stems' / (name + '.wav'), track / 'MIDI' / (name + '.mid')
            if stem['is_drum'] or 32 <= stem['program_num'] < 40:
                continue
            if not audio.is_file() or not midi.is_file():
                continue
            actual = sf.info(audio)
            if actual.samplerate != info.samplerate or actual.channels != 1 or actual.frames != info.frames:
                raise ValueError('Source stem and mixture clocks differ')
            source_keys, source_support = intervals(pretty_midi.PrettyMIDI(str(midi)), info.duration)
            selected.append(audio)
            keys.extend(source_keys.tolist())
            support.extend(source_support.tolist())
            audit.append({'track': identity, 'source': name, 'class': stem['inst_class'],
                          'audio_sha256': digest(audio), 'midi_sha256': digest(midi),
                          'metadata_audio_rendered': stem['audio_rendered'], 'actual_source_files_present': True,
                          'note_labels': len(source_keys)})
        if not selected:
            raise ValueError('No aligned pitched source pairs')
        keys = np.asarray(sorted(set(map(tuple, keys))), dtype=float).reshape(-1, 3)
        support = np.asarray(sorted(set(map(tuple, support))), dtype=float).reshape(-1, 3)
        for index, offset in enumerate(plan['offsets']):
            identity_clip = f'slakh-v1-{identity}-{offset:03}'
            work = output / identity_clip
            work.mkdir(exist_ok=True)
            completed = work / 'complete.json'
            if completed.exists():
                record = json.loads(completed.read_text())
                cache = directory / 'note-verifier-context-expanded/note-verifier-features' / (identity_clip + '.npz')
                if (record['plan_sha256'] != digest(output / 'plan.json')
                        or record['cache_sha256'] != digest(cache)
                        or record['audio_sha256'] != digest(work / 'decoded.wav')):
                    raise ValueError('Changed completed prototype preparation')
                item = record['item']
            else:
                count = 30 * info.samplerate
                samples = np.zeros(count, dtype=np.float32)
                for audio in selected:
                    piece, rate = sf.read(audio, start=offset * info.samplerate, frames=count, dtype='float32')
                    if rate != info.samplerate or len(piece) != count or not np.isfinite(piece).all():
                        raise ValueError('Source waveform does not fit the fixed clip clock')
                    samples += piece
                samples *= .95 / max(.95, float(np.max(np.abs(samples))))
                samples = librosa.resample(samples, orig_sr=info.samplerate, target_sr=22050)
                sf.write(work / 'submix.wav', samples, 22050, subtype='FLOAT')
                bitrate = plan['mp3_bitrates'][index % 3]
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'submix.wav'),
                                '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k', str(work / 'submix.mp3')], check=True)
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'submix.mp3'),
                                '-ac', '1', '-ar', '22050', '-c:a', 'pcm_f32le', str(work / 'decoded.wav')], check=True)
                decoded, _ = sf.read(work / 'decoded.wav', dtype='float32')
                clock = verify_codec(samples, decoded, 22050)
                duration = min(len(samples), len(decoded)) / 22050
                reference, pitch = crop(keys, support, offset, duration)
                item = {'id': identity_clip, 'corpus': 'slakh-oracle-accompaniment', 'group': groups[identity],
                        'source_group': duplicates[identity]['midi_md5'],
                        'audio': str((work / 'decoded.wav').resolve()), 'reference': reference.tolist(),
                        'pitch_reference': pitch.tolist(), 'evaluation_window': [.25, duration - .25],
                        'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True}
                cache_one((str(directory / 'note-verifier-context-expanded'), item))
                cache = directory / 'note-verifier-context-expanded/note-verifier-features' / (identity_clip + '.npz')
                record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                          'cache_sha256': digest(cache), 'audio_sha256': digest(work / 'decoded.wav'),
                          'codec_clock': clock, 'bitrate_kbps': bitrate}
                preserve(completed, record)
            items.append(item)
            print(json.dumps({'cached': len(items), 'total': 64, 'id': identity_clip, 'group': groups[identity]}), flush=True)
    preserve(output / 'source-audit.json', {'items': audit,
             'flags': 'Publisher prototype rendered flags are false despite actual paired files. Actual audio/MIDI clock and presence checked; no model-based label timing.'})
    preserve(output / 'manifest.json', {'items': items, 'plan_sha256': digest(output / 'plan.json'),
             'scope': plan['scope'], 'no_test_inference': True, 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
