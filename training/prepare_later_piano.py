"""Prepare unscored later piano sections without consulting model predictions."""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


def prepare(directory):
    output = directory / 'later-vienna-context'
    output.mkdir(exist_ok=False)
    clocks = json.loads((directory / 'note-verifier-v3-data/clock-audit.json').read_text())
    items = []
    for clock in clocks:
        if clock['group'] != 'test':
            continue
        name = clock['id'].removeprefix('vienna-')
        matches = list((directory / 'vienna/audio').rglob(name + '.wav'))
        if len(matches) != 1:
            raise ValueError(f'Ambiguous audio source: {name}')
        samples, rate = librosa.load(matches[0], sr=22050, offset=30, duration=30)
        if len(samples) / rate < 5:
            raise ValueError(f'Insufficient later recording: {name}')
        sf.write(output / (name + '.wav'), samples, rate, subtype='FLOAT')
        midi = pretty_midi.PrettyMIDI(str(directory / 'vienna/midi' / (name + '.mid')))
        reference = np.asarray([[note.start, note.end, note.pitch] for part in midi.instruments for note in part.notes])
        reference[:, :2] += clock['initial_shift'] + clock['spectral_correction'] - 30
        stop = len(samples) / rate - .25
        reference = reference[(reference[:, 0] >= .25) & (reference[:, 0] < stop)]
        reference[:, 1] = np.minimum(reference[:, 1], len(samples) / rate)
        if not len(reference):
            raise ValueError(f'No later reference notes: {name}')
        items.append({'id': 'later-' + name, 'corpus': 'later-vienna-piano',
            'audio': '../later-vienna-context/' + name + '.wav',
            'reference': reference.tolist(), 'evaluation_window': [.25, stop],
            'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True})
    (output / 'manifest.json').write_text(json.dumps({'items': items,
        'scope': 'Previously unscored 30–60 second sections of consumed test recordings; new notes, not new performers or compositions',
        'clock': 'Frozen first-15-second acoustic calibration; no predictions used',
        'gates': 'Same release gate against original detector and deployed V2; report every recording'}, indent=2) + '\n')
    print(json.dumps({'prepared': len(items)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
