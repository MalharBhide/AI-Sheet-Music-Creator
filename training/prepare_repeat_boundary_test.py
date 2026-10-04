"""Prepare a frozen held-out MP3/separation boundary stress test after regression.

Sources and conditions are selected by fixed order, never prediction quality.
Existing test performer recordings are reused under previously unscored codec
and separation conditions. This is not an independent recording benchmark.
"""

import argparse
import contextlib
import json
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from app.services.source_separation import StemSeparator
from cache_note_verifier import cache_one
from prepare_robust_training_stems import codec_clock, digest, preserve
from prepare_verifier_tests import percussion
from repeat_boundary_release import load_frozen, require_report

SEED = 261031


def select(manifest):
    selected = []
    for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS'):
        for role in ('comp', 'solo'):
            choices = sorted((item for item in manifest['tracks']['test']
                if item['corpus'] == 'guitarset' and item['id'].startswith(f'05_{genre}')
                and item['id'].endswith(role)), key=lambda item: item['id'])
            if len(choices) < 2 or choices[1]['player'] != '05':
                raise ValueError('Missing or mismatched reserved test performer')
            selected.append({**choices[1], 'group': 'test'})
    return selected


def prepare(directory, run):
    winner, _ = load_frozen(run)
    require_report(run, winner, 'regression', positive=True)
    require_report(run, winner, 'current-piano-regression')
    base = directory / 'note-verifier-v3-data'
    manifest = base / 'note-verifier-split.json'
    selected = select(json.loads(manifest.read_text()))
    output = directory / 'repeat-boundary-stress-v1'
    output.mkdir(exist_ok=True)
    plan = {'seed': SEED, 'manifest_sha256': digest(manifest),
            'frozen_candidate_sha256': winner['checkpoint_sha256'],
            'batch_selection_sha256': digest(run / 'batch-selection.json'),
            'preparation_sha256': digest(Path(__file__)),
            'codec_code_sha256': digest(Path(__file__).with_name('prepare_robust_training_stems.py')),
            'cache_code_sha256': digest(Path(__file__).with_name('cache_note_verifier.py')),
            'separator_code_sha256': digest(Path(__file__).resolve().parents[1] / 'backend/app/services/source_separation.py'),
            'sources': [{'id': item['id'], 'group': 'test', 'audio_sha256': digest(base / item['audio'])}
                        for item in selected],
            'backing_db_cycle': [-6, 0, 3], 'mp3_bitrate_kbps_cycle': [48, 128, 192],
            'selection': 'Second sorted comp/solo in each genre, reserved performer 05 only',
            'scope': 'Unscored codec/separation conditions on previously seen test sources; no training or selection',
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
                raise ValueError('Changed frozen stress preparation')
            item = record['item']
            cache = directory / 'note-verifier-context-expanded/note-verifier-features' / (item['id'] + '.npz')
            if digest(cache) != record['cache_sha256'] or digest(work / 'other.wav') != record['audio_sha256']:
                raise ValueError('Changed completed stress evidence')
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
                raise ValueError('No usable accompaniment stem for the frozen stress source')
            duration = sf.info(work / 'other.wav').duration
            stop = min(len(samples) / rate, duration) - .25
            reference = [[start, min(end, duration), pitch] for start, end, pitch in source['reference']
                         if .25 <= start < stop and end > start]
            if not reference:
                raise ValueError('No original reference notes in stress source')
            item = {**source, 'id': 'boundary-stress-v1-' + source['id'], 'corpus': 'separated-guitar',
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
        print(json.dumps({'completed': index + 1, 'total': len(selected),
                          'source': source['id'], 'codec_lag_seconds': record['codec_clock']['codec_lag_seconds']}), flush=True)
    preserve(output / 'manifest.json', {'items': items, 'plan_sha256': digest(output / 'plan.json'),
             'scope': plan['scope'], 'no_user_audio_or_scores': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.run)
