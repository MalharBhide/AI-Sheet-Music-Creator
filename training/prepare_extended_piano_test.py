"""Prepare previously unscored 30–60s piano segments after checkpoint freeze."""

import argparse
import json
import subprocess
from pathlib import Path

import pretty_midi


def prepare(directory):
    output = directory / 'note-verifier-oxford-extended'
    output.mkdir(exist_ok=True)
    clocks = json.loads((directory / 'note-verifier-oxford/manifest.json').read_text())['clock']
    source = directory / 'oxford-miditest/MIDItest'
    items = []
    for clock in clocks:
        identity = clock['id']
        path = output / (identity + '.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source / 'miditest_videos' / (identity + '.mp4')),
                        '-ss', '30', '-t', '30', '-vn', '-ar', '22050', '-ac', '1', str(path)], check=True)
        midi = pretty_midi.PrettyMIDI(str(source / 'miditest_MIDI' / (identity + '.mid')))
        shift = clock['shift_seconds'] - 30
        reference = [[n.start + shift, min(30., n.end + shift), n.pitch]
                     for part in midi.instruments for n in part.notes if .25 <= n.start + shift < 29.75]
        items.append({'id': 'oxford-extended-' + identity, 'corpus': 'oxford-piano-extended',
                      'audio': str(path.relative_to(directory)), 'reference': reference,
                      'evaluation_window': [.25, 29.75]})
    (output / 'manifest.json').write_text(json.dumps({'items': items,
        'scope': 'Previously unscored 30–60s; same performers/recordings as the consumed first test, not independent recordings',
        'clock_source': 'Unchanged 0–15s acoustic clock calibration'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
