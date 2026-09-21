"""Add source-separated examples from training performers only."""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from prepare_verifier_tests import percussion

from app.services.source_separation import StemSeparator


def prepare(directory):
    manifest = json.loads((directory / 'note-verifier-v3-data/note-verifier-split.json').read_text())
    training = manifest['tracks']['train']
    selected = [next(item for item in training if item['corpus'] == 'guitarset'
                     and item['id'].startswith(f'{player:02}_{genre}') and item['id'].endswith(role))
                for player in range(4) for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS')
                for role in ('comp', 'solo')]
    output = directory / 'note-verifier-training-stems'
    output.mkdir(exist_ok=True)
    plan = {'seed': 260921, 'ids': [item['id'] for item in selected],
            'backing_db': -3, 'selection': 'First comp/solo per genre for training performers 00–03 only'}
    plan_path = output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Do not overwrite a different training separation plan')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    separator, items = StemSeparator(), []
    for index, item in enumerate(selected):
        started = time.monotonic()
        work = output / item['id']
        work.mkdir(exist_ok=True)
        if not (work / 'complete.json').exists():
            source = directory / 'note-verifier-v3-data' / item['audio']
            samples, rate = librosa.load(source, sr=22050, duration=30)
            backing = percussion(len(samples), rate, plan['seed'] + index)
            backing *= np.sqrt(np.mean(samples ** 2)) / max(1e-8, np.sqrt(np.mean(backing ** 2))) * 10 ** (-3 / 20)
            mixture = samples + backing
            mixture *= .95 / max(.95, float(np.max(np.abs(mixture))))
            sf.write(work / 'mixture.wav', mixture, rate, subtype='FLOAT')
            with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
                separated = separator.separate(work / 'mixture.wav', work)
            if 'other' not in separated:
                sf.write(work / 'other.wav', np.zeros_like(samples), rate, subtype='FLOAT')
            (work / 'complete.json').write_text(json.dumps({'source': item['id'], 'plan': plan}) + '\n')
        items.append({**item, 'id': 'training-stem-' + item['id'], 'corpus': 'separated-guitar',
                      'audio': '../' + str((work / 'other.wav').relative_to(directory)),
                      'context_features': True})
        print(json.dumps({'completed': index + 1, 'total': len(selected), 'id': item['id'],
                          'seconds': round(time.monotonic() - started, 2)}), flush=True)
    (output / 'manifest.json').write_text(json.dumps({'plan': plan, 'items': items}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
