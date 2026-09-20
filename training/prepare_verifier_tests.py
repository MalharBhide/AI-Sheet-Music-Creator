"""Prepare independent/stem tests without examining verifier predictions."""

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


def align_clock(samples, rate, reference):
    """Estimate a single label-clock shift on the first 15 seconds only.

    No model predictions enter alignment. Score on the disjoint second half.
    Candidate shifts and spectral-onset calculation are fixed for all tracks.
    """
    hop = 128
    spectrum = np.abs(librosa.cqt(samples[:15 * rate], sr=rate, hop_length=hop,
                                  fmin=librosa.midi_to_hz(21), n_bins=88))
    log = np.log1p(spectrum * 10)
    flux = np.maximum(0, np.diff(log, axis=1, prepend=log[:, :1]))
    flux /= np.maximum(.001, flux.max(axis=1, keepdims=True))
    onsets = reference[(reference[:, 0] >= 1) & (reference[:, 0] < 14)]
    shifts = np.arange(-.6, .201, .005)
    scores = []
    for shift in shifts:
        values = []
        for start, _, pitch in onsets:
            center = round((start + shift) * rate / hop)
            # A narrow window covers CQT temporal smearing without a learned offset.
            values.append(flux[int(pitch) - 21, max(0, center - 2):center + 3].max())
        scores.append(float(np.mean(values)))
    best = int(np.argmax(scores))
    return float(shifts[best]), {'shift_seconds': float(shifts[best]), 'calibration_onsets': len(onsets),
                                'score': scores[best], 'boundary': best in (0, len(shifts) - 1)}


def oxford(directory):
    root = directory / 'oxford-miditest/MIDItest'
    output = directory / 'note-verifier-oxford'
    output.mkdir(exist_ok=True)
    items, audit = [], []
    for midi in sorted((root / 'miditest_MIDI').glob('*.mid')):
        path = output / (midi.stem + '.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(root / 'miditest_videos' / (midi.stem + '.mp4')),
                        '-vn', '-t', '30', '-ar', '22050', '-ac', '1', str(path)], check=True)
        samples, rate = sf.read(path, dtype='float32')
        reference = np.asarray([[n.start, n.end, n.pitch] for part in pretty_midi.PrettyMIDI(str(midi)).instruments for n in part.notes])
        shift, record = align_clock(samples, rate, reference)
        if record['boundary']:
            raise ValueError(f'Clock alignment hit search boundary: {midi.stem}')
        reference[:, :2] += shift
        # Exclude calibration, crop-edge attacks and their context from scoring.
        reference = reference[(reference[:, 0] >= 15.25) & (reference[:, 0] < 29.75)]
        reference[:, 1] = np.minimum(reference[:, 1], 30)
        items.append({'id': 'oxford-' + midi.stem, 'corpus': 'oxford-piano',
                      'audio': str(path.relative_to(directory)), 'reference': reference.tolist(),
                      'evaluation_window': [15.25, 29.75]})
        audit.append({'id': midi.stem, **record})
    manifest = {'items': items, 'clock': audit,
                'alignment': 'Fixed CQT onset alignment on 0–15s, shift -0.6..0.2s at 5ms resolution; evaluate only 15.25–29.75s. No transcription predictions used.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'clock_audit': audit}), flush=True)


def percussion(length, rate, seed):
    rng = np.random.default_rng(seed)
    result = np.zeros(length, dtype=np.float32)
    for index, start in enumerate(np.arange(.25, length / rate, .25)):
        size = int(rate * (.12 if index % 4 == 0 else .05))
        t = np.arange(size) / rate
        sound = rng.normal(size=size) * np.exp(-t / (.025 if index % 4 == 0 else .008))
        first = int(start * rate)
        count = min(size, length - first)
        result[first:first + count] += sound[:count]
    return result


def stems(directory, group):
    from app.services.source_separation import StemSeparator

    manifest = json.loads((directory / 'note-verifier-split.json').read_text())
    # Fixed, genre-spread first comp/solo pair in each genre for validation/test.
    selected = []
    for genre in ('BN', 'Funk', 'Jazz', 'Rock', 'SS'):
        for role in ('comp', 'solo'):
            selected.append(next(i for i in manifest['tracks'][group]
                                 if i['corpus'] == 'guitarset' and i['id'][3:].startswith(genre)
                                 and i['id'].endswith(role)))
    output = directory / ('note-verifier-stems-' + group)
    output.mkdir(exist_ok=True)
    separator, items = StemSeparator(), []
    for index, item in enumerate(selected):
        work = output / item['id']
        work.mkdir(exist_ok=True)
        samples, rate = librosa.load(directory / item['audio'], sr=22050, duration=30)
        backing = percussion(len(samples), rate, SEED + index)
        backing *= np.sqrt(np.mean(samples ** 2)) / max(1e-8, np.sqrt(np.mean(backing ** 2))) * 10 ** (-3 / 20)
        mixture = samples + backing
        mixture *= .95 / max(.95, float(np.max(np.abs(mixture))))
        sf.write(work / 'mixture.wav', mixture, rate, subtype='FLOAT')
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            separated = separator.separate(work / 'mixture.wav', work)
        if 'other' not in separated:
            sf.write(work / 'other.wav', np.zeros_like(samples), rate, subtype='FLOAT')
        items.append({**item, 'id': 'stem-' + item['id'], 'corpus': 'separated-guitar',
                      'audio': str((work / 'other.wav').relative_to(directory))})
        print(json.dumps({'prepared_stem': item['id']}), flush=True)
    (output / 'manifest.json').write_text(json.dumps({'items': items, 'backing_db': -3,
        'selection': 'First comp/solo example in each of five genres; unpitched procedural percussion only',
        'seed': SEED}, indent=2) + '\n')


SEED = 260920

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stems', choices=['validation', 'test'])
    args = parser.parse_args()
    if args.stems:
        stems(args.directory, args.stems)
    else:
        oxford(args.directory)
