"""Original piano phrases covering quiet attacks, repeated keys and dense chords."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pretty_midi

from app.config import Settings
from app.models import ScoreOptions
from app.services.midi_to_score import midi_to_musicxml
from app.services.playback import export_score_playback


def render(task):
    directory, index = task
    root = Path(directory) / 'context-piano'
    name = f'context-piano-{index:03}'
    root = root / name
    root.mkdir(exist_ok=True)
    midi, xml, wav, labels = [root / (name + ext) for ext in ('.mid', '.musicxml', '.wav', '.json')]
    bpm = (96, 112, 128, 144, 160, 176)[index % 6]
    beat = 60 / bpm
    if not wav.exists() or not labels.exists():
        rng = np.random.default_rng(260921 + index)
        part = pretty_midi.Instrument(0)
        key = int(rng.integers(-6, 6))
        for bar in range(6):
            root_pitch = int(rng.choice([40, 43, 45, 47, 48, 50])) + key
            quality = rng.choice([[0, 3, 7], [0, 4, 7], [0, 5, 9], [0, 4, 10]])
            start = (1 + bar * 4) * beat
            # Independent hands, including quiet repeated accompaniment under
            # louder chords; all pieces are generated, not copied from test MIDI.
            for pulse in range(8):
                onset = start + pulse * .5 * beat
                if pulse % 2 == 0:
                    for interval in quality:
                        pitch = root_pitch + int(interval)
                        velocity = int(rng.integers(22, 65) if index % 2 else rng.integers(65, 110))
                        part.notes.append(pretty_midi.Note(velocity, pitch, onset,
                            onset + float(rng.choice([.25, .5, 1.])) * beat))
                if rng.random() > .08:
                    interval = int(quality[(pulse + bar) % 3]) if index % 3 else int(rng.integers(0, 12))
                    pitch = root_pitch + 24 + interval
                    velocity = int(rng.integers(25, 60) if index % 2 == 0 else rng.integers(70, 115))
                    part.notes.append(pretty_midi.Note(velocity, pitch, onset,
                        onset + float(rng.choice([.25, .5, .75])) * beat))
                if index % 4 == 0 and pulse % 2 == 0:
                    part.notes.append(pretty_midi.Note(50, root_pitch + 12, onset, onset + beat))
        # Never label two simultaneous independent strikes of a single piano key.
        by_pitch = {}
        for item in sorted(part.notes, key=lambda n: (n.start, n.pitch)):
            if item.pitch in by_pitch and by_pitch[item.pitch].end > item.start:
                by_pitch[item.pitch].end = item.start
            by_pitch[item.pitch] = item
        part.notes = [n for n in part.notes if n.end > n.start]
        source = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        source.instruments.append(part)
        source.write(str(midi))
        midi_to_musicxml(midi, xml, ScoreOptions(tempo_bpm=bpm, grid='sixteenth'), 'Original context training phrase')
        export_score_playback(xml, midi, labels, bpm)
        result = subprocess.run([Settings(_env_file=None).renderer(), '-o', str(wav), str(xml)],
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    reference = json.loads(labels.read_text())
    return {'id': name, 'corpus': 'original-piano', 'audio': f'../context-piano/{name}/{name}.wav',
            'reference': [[n['start'], n['end'], n['pitch']] for n in reference['notes']],
            'candidate_decoder': 'bounded-accompaniment-v1', 'context_features': True,
            'group': 'train' if index < 64 else 'validation'}


def prepare(directory, workers):
    root = directory / 'context-piano'
    root.mkdir(exist_ok=True)
    items = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, item in enumerate(pool.map(render, [(str(directory), n) for n in range(80)])):
            items.append(item)
            print(json.dumps({'rendered': index + 1, 'total': 80, 'id': item['id']}), flush=True)
    (root / 'manifest.json').write_text(json.dumps({'seed': 260921, 'items': items,
        'rights': 'Original generated compositions; installed MuseScore General MIT/PD/CC0 soundfont. Synthetic, not evidence of real-performance accuracy.'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    prepare(args.directory, args.workers)
