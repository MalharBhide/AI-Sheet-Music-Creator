"""Original grid-aligned piano phrases for training; no existing compositions."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi

from app.config import Settings
from app.models import ScoreOptions
from app.services.midi_to_score import midi_to_musicxml
from app.services.playback import export_score_playback


def generate(directory):
    root = directory / 'verifier-piano'
    root.mkdir(exist_ok=True)
    settings = Settings(_env_file=None)
    for index in range(40):
        name = f'piano-{index:02}'
        midi, xml, wav = [root / (name + extension) for extension in ('.mid', '.musicxml', '.wav')]
        reference = root / (name + '.json')
        if wav.exists() and reference.exists():
            continue
        rng = np.random.default_rng(260920 + index)
        bpm = [80, 100, 120, 160][index % 4]
        beat = 60 / bpm
        source = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        piano = pretty_midi.Instrument(0)
        # Chords, separated attacks, repeated keys, silence, low notes and
        # chromatic notes. The model cannot learn that out-of-key means wrong.
        transpose = int(rng.integers(-5, 6))
        for bar in range(4):
            root_pitch = int(rng.choice([36, 41, 43, 45])) + transpose
            onset = .5 * beat + bar * 4 * beat
            for interval in rng.choice([[0, 16, 19], [0, 15, 19], [0, 19, 26], [0, 14, 21]]):
                piano.notes.append(pretty_midi.Note(int(rng.integers(45, 100)), root_pitch + int(interval),
                                                    onset, onset + beat * float(rng.choice([1, 2, 3]))))
            for pulse in range(8):
                if rng.random() < .22:
                    continue
                pitch = 60 + transpose + int(rng.integers(0, 18))
                start = onset + pulse * .5 * beat
                end = start + float(rng.choice([.25, .5])) * beat
                piano.notes.append(pretty_midi.Note(int(rng.integers(45, 110)), pitch, start, end))
        source.instruments.append(piano)
        source.write(str(midi))
        midi_to_musicxml(midi, xml, ScoreOptions(tempo_bpm=bpm), 'Original training phrase')
        export_score_playback(xml, midi, reference, bpm)
        result = subprocess.run([settings.renderer(), '-o', str(wav), str(xml)],
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
        print(json.dumps({'rendered': name}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    generate(parser.parse_args().directory)
