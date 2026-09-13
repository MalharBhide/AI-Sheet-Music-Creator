"""Frozen-model stress test: real held-out vocals, procedural backing, Demucs.

This is a controlled mixture test, not an accuracy benchmark of commercial songs.
Backing is generated independently of note annotations. Never use these test
outputs for checkpoint or decoder selection.
"""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model
from basic_pitch.inference import predict as acoustic_predict
from train_melody import evaluate, predict

from app.services.melody_decoder import MelodyDecoder, features
from app.services.source_separation import StemSeparator
from app.services.vocal_refinement import refine_notes


def backing(length, rate, seed):
    """Deterministic original bass/chords/percussion, unrelated to vocal labels."""
    rng = np.random.default_rng(seed)
    output = np.zeros(length, dtype=np.float32)
    for onset in np.arange(0, length / rate, .5):
        root = [48, 45, 53, 55][int(onset // 2) % 4]
        time = np.arange(min(round(.45 * rate), length - round(onset * rate))) / rate
        signal = np.zeros(len(time))
        for pitch, amplitude in [(root - 12, .4), (root, .2), (root + 4, .15), (root + 7, .15)]:
            frequency = float(librosa.midi_to_hz(pitch))
            signal += amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-time * 7)
        signal += .08 * rng.standard_normal(len(time)) * np.exp(-time * 50)
        start = round(onset * rate)
        output[start:start + len(time)] += signal
    return output


def run(directory):
    torch.set_num_threads(2)
    run_path = directory / 'melody-v1'
    if not (run_path / 'test.json').exists():
        raise ValueError('Freeze and evaluate the clean test first')
    target = run_path / 'separated-test.json'
    if target.exists():
        raise ValueError('Separated test already evaluated; do not tune against it')
    selection = json.loads((run_path / 'selection.json').read_text())
    digest = hashlib.sha256((run_path / 'candidate.pt').read_bytes()).hexdigest()
    if digest != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    split = json.loads((run_path / 'split.json').read_text())
    checkpoint = torch.load(run_path / 'candidate.pt', weights_only=True, map_location='cpu')
    model = MelodyDecoder()
    model.load_state_dict(checkpoint['state_dict'])
    separator, acoustic = StemSeparator(), Model(ICASSP_2022_MODEL_PATH)
    dataset = []
    for track in split['tracks']['test']:
        work = directory / 'stress' / str(track)
        work.mkdir(parents=True, exist_ok=True)
        voice, rate = librosa.load(directory / 'vocadito' / 'Audio' / f'vocadito_{track}.wav', sr=22050)
        support = backing(len(voice), rate, 1729 + track)
        # Backing 3 dB below the complete vocal clip's RMS (including its rests).
        support *= np.sqrt(np.mean(voice ** 2)) / max(1e-8, np.sqrt(np.mean(support ** 2))) * 10 ** (-3 / 20)
        mixture = voice + support
        mixture *= .95 / max(.95, float(np.max(np.abs(mixture))))
        sf.write(work / 'mixture.wav', mixture, rate, subtype='FLOAT')
        stems = separator.separate(work / 'mixture.wav', work)
        if 'vocals' not in stems:
            sf.write(work / 'vocals.wav', np.zeros_like(voice), rate, subtype='FLOAT')
        path = work / 'vocals.wav'
        arrays, midi, _ = acoustic_predict(str(path), model_or_model_path=acoustic,
            minimum_frequency=float(librosa.midi_to_hz(45)), maximum_frequency=float(librosa.midi_to_hz(96)),
            onset_threshold=.5, frame_threshold=.3, minimum_note_length=90,
            multiple_pitch_bends=False, melodia_trick=False, midi_tempo=120)
        samples, rate = sf.read(path, dtype='float32')
        x, (pitches, confidence, rms) = features(samples, rate, arrays)
        notes = [note for part in midi.instruments
                 for note in refine_notes(part.notes, pitches, confidence, 256 / rate, rms)]
        baseline = np.asarray([[n.start, n.end, n.pitch] for n in notes]).reshape(-1, 3)
        with np.load(directory / 'features' / f'vocadito_{track}.npz') as cached:
            a1, a2 = cached['A1'], cached['A2']
        dataset.append({'id': track, 'x': x, 'baseline': baseline, 'A1': a1, 'A2': a2})
        np.savez_compressed(work / 'features.npz', x=x, baseline=baseline, A1=a1, A2=a2)
        print(json.dumps({'separated': track}), flush=True)
    outputs = predict(model, dataset)
    report = {'checkpoint_sha256': digest, 'backing_level_db': -3,
              'description': 'Real held-out singing with procedural backing, passed through Demucs htdemucs',
              'A1': evaluate(dataset, outputs, checkpoint['decoder'], 'A1'),
              'A2': evaluate(dataset, outputs, checkpoint['decoder'], 'A2')}
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key]['aggregate'] for key in ['A1', 'A2']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
