"""Prepare unscored 60–90s sections, keeping the original audio-only clocks."""

import argparse
import json
import subprocess
from pathlib import Path

import pretty_midi
import soundfile as sf


def prepare(directory):
    output = directory / 'oxford-later-context'
    output.mkdir(exist_ok=False)
    clocks = json.loads((directory / 'note-verifier-oxford/manifest.json').read_text())['clock']
    source = directory / 'oxford-miditest/MIDItest'
    items = []
    for clock in clocks:
        identity = clock['id']
        path = output / (identity + '.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i',
                        str(source / 'miditest_videos' / (identity + '.mp4')),
                        '-ss', '60', '-t', '30', '-vn', '-ar', '22050', '-ac', '1', str(path)], check=True)
        duration = sf.info(path).duration
        if duration < 5:
            raise ValueError(f'Insufficient later audio: {identity}')
        midi = pretty_midi.PrettyMIDI(str(source / 'miditest_MIDI' / (identity + '.mid')))
        shift = clock['shift_seconds'] - 60
        stop = duration - .25
        reference = [[n.start + shift, min(duration, n.end + shift), n.pitch]
                     for part in midi.instruments for n in part.notes if .25 <= n.start + shift < stop]
        if not reference:
            raise ValueError(f'No later reference notes: {identity}')
        items.append({'id': 'oxford-later-' + identity, 'corpus': 'oxford-later-piano',
                      'audio': '../oxford-later-context/' + identity + '.wav',
                      'reference': reference, 'evaluation_window': [.25, stop]})
    (output / 'manifest.json').write_text(json.dumps({'items': items,
        'scope': 'Previously unscored 60–90s (or remaining audio); same recordings, not new performers/compositions',
        'clock': 'Unchanged 0–15s audio-only calibration; those calibration notes are excluded'}, indent=2) + '\n')
    print(json.dumps({'prepared': len(items)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
