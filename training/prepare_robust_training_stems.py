"""Cache MP3/percussion separation augmentation from licensed GuitarSet only.

Fixed training/validation performers; no test audio or score generation. This
resumable preparation never changes the website or overwrites completed evidence.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from app.services.source_separation import StemSeparator
from cache_note_verifier import cache_one
from prepare_verifier_tests import percussion
from scipy.signal import correlate, correlation_lags

SEED = 261024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(path, content):
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError('Do not overwrite different completed preparation evidence')
    else:
        path.write_text(json.dumps(content, indent=2) + '\n')


def select(manifest):
    selected = []
    for group, players in [('train', range(4)), ('validation', [4])]:
        for player in players:
            for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS'):
                for role in ('comp', 'solo'):
                    choices = sorted((item for item in manifest['tracks'][group]
                        if item['corpus'] == 'guitarset'
                        and item['id'].startswith(f'{player:02}_{genre}') and item['id'].endswith(role)),
                        key=lambda item: item['id'])
                    if len(choices) < 2:
                        raise ValueError('Missing second labeled source for frozen augmentation plan')
                    item = choices[1]
                    if int(item['player']) != player:
                        raise ValueError('Source performer does not match its dataset split')
                    selected.append({**item, 'group': group})
    return selected


def codec_clock(original, decoded, rate):
    """Reject encoder delay rather than learning from shifted annotations."""
    if abs(len(original) - len(decoded)) > .03 * rate:
        raise ValueError('MP3 gapless duration check failed')
    count = min(len(original), len(decoded), 5 * rate)
    if count < rate or np.std(original[:count]) < 1e-6:
        raise ValueError('Insufficient waveform for MP3 clock verification')
    correlation = correlate(decoded[:count], original[:count], method='fft')
    lags = correlation_lags(count, count)
    bounded = np.abs(lags) <= round(.05 * rate)
    lag = int(lags[bounded][np.argmax(correlation[bounded])])
    if abs(lag) > .01 * rate:
        raise ValueError('MP3 waveform clock drift; labels must not be shifted from predictions')
    return {'codec_lag_samples': lag, 'codec_lag_seconds': lag / rate,
            'input_frames': len(original), 'decoded_frames': len(decoded)}


def prepare(directory):
    base = directory / 'note-verifier-v3-data'
    manifest_path = base / 'note-verifier-split.json'
    selected = select(json.loads(manifest_path.read_text()))
    output = directory / 'robust-training-stems-v1'
    output.mkdir(exist_ok=True)
    plan = {'seed': SEED, 'manifest_sha256': digest(manifest_path),
            'sources': [{'id': item['id'], 'group': item['group'],
                         'audio_sha256': digest(base / item['audio'])} for item in selected],
            'preparation_sha256': digest(Path(__file__)),
            'cache_code_sha256': digest(Path(__file__).with_name('cache_note_verifier.py')),
            'separator_code_sha256': digest(Path(__file__).resolve().parents[1] / 'backend/app/services/source_separation.py'),
            'backing_db_cycle': [-6, 0, 3], 'mp3_bitrate_kbps_cycle': [48, 128, 192],
            'selection': 'Second sorted comp/solo per genre; performers 00-03 train, 04 validation; no test performers',
            'labels': 'Original GuitarSet notes, clipped to audio/crop; procedural backing is unpitched',
            'scope': 'Training augmentation, shared pieces with raw sources; not independent evaluation',
            'no_user_audio_or_scores': True}
    preserve(output / 'plan.json', plan)
    separator, items = None, []
    for index, source in enumerate(selected):
        work = output / source['id']
        work.mkdir(exist_ok=True)
        completed = work / 'complete.json'
        if completed.exists():
            record = json.loads(completed.read_text())
            if record['plan_sha256'] != digest(output / 'plan.json'):
                raise ValueError('Changed frozen augmentation plan')
            item = record['item']
            cache = directory / 'note-verifier-context-expanded/note-verifier-features' / (item['id'] + '.npz')
            if digest(cache) != record['cache_sha256'] or digest(work / 'other.wav') != record['audio_sha256']:
                raise ValueError('Changed completed augmentation audio/features')
        else:
            samples, rate = librosa.load(base / source['audio'], sr=22050, duration=30)
            db, bitrate = plan['backing_db_cycle'][index % 3], plan['mp3_bitrate_kbps_cycle'][index % 3]
            backing = percussion(len(samples), rate, SEED + index)
            backing *= np.sqrt(np.mean(samples ** 2)) / max(1e-8, np.sqrt(np.mean(backing ** 2))) * 10 ** (db / 20)
            mixture = samples + backing
            mixture *= .95 / max(.95, float(np.max(np.abs(mixture))))
            sf.write(work / 'uncompressed.wav', mixture, rate, subtype='FLOAT')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'uncompressed.wav'),
                            '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k', str(work / 'mixture.mp3')], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'mixture.mp3'),
                            '-ar', str(rate), '-ac', '1', '-c:a', 'pcm_f32le', str(work / 'decoded.wav')], check=True)
            decoded, _ = sf.read(work / 'decoded.wav', dtype='float32')
            clock = codec_clock(mixture, decoded, rate)
            if separator is None:
                separator = StemSeparator()
            with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                paths = separator.separate(work / 'decoded.wav', work)
            if 'other' not in paths:
                sf.write(work / 'other.wav', np.zeros_like(decoded), rate, subtype='FLOAT')
            duration = sf.info(work / 'other.wav').duration
            stop = min(len(samples) / rate, duration) - .25
            reference = [[start, min(end, duration), pitch] for start, end, pitch in source['reference']
                         if .25 <= start < stop and end > start]
            if not reference:
                raise ValueError('No reference notes in augmented source')
            item = {**source, 'id': 'robust-stem-v1-' + source['id'], 'corpus': 'separated-guitar',
                    'audio': str((work / 'other.wav').resolve()), 'reference': reference,
                    'evaluation_window': [.25, stop], 'candidate_decoder': 'bounded-accompaniment-v1',
                    'context_features': True}
            cache_one((str(directory / 'note-verifier-context-expanded'), item))
            cache = directory / 'note-verifier-context-expanded/note-verifier-features' / (item['id'] + '.npz')
            record = {'item': item, 'plan_sha256': digest(output / 'plan.json'),
                      'audio_sha256': digest(work / 'other.wav'), 'cache_sha256': digest(cache),
                      'codec_clock': clock, 'backing_db': db, 'bitrate_kbps': bitrate}
            preserve(completed, record)
        items.append(item)
        print(json.dumps({'completed': index + 1, 'total': len(selected), 'group': source['group'],
                          'source': source['id'], 'codec_lag_seconds': record['codec_clock']['codec_lag_seconds']}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'plan_sha256': digest(output / 'plan.json'),
             'scope': plan['scope'], 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
