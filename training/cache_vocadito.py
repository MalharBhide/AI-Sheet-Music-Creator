"""Cache frozen features and the existing detector's predictions, not fitted labels."""

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path

import librosa
import numpy as np
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

from app.services.melody_decoder import FEATURE_VERSION, features
from app.services.vocal_refinement import refine_notes


def run(directory):
    corpus = directory / 'vocadito'
    cache = directory / 'features'
    cache.mkdir(exist_ok=True)
    model = Model(ICASSP_2022_MODEL_PATH)
    rows = list(csv.DictReader((corpus / 'vocadito_metadata.csv').open()))
    for row in rows:
        name = 'vocadito_' + row['track_id']
        target = cache / (name + '.npz')
        if target.exists():
            with np.load(target) as previous:
                if str(previous['version']) == FEATURE_VERSION:
                    continue
        samples, rate = librosa.load(corpus / 'Audio' / (name + '.wav'), sr=22050, mono=True)
        # Use the exact deployed Basic Pitch thresholds and vocal pitch range.
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            import soundfile as sf
            normalized = directory / 'normalized.wav'
            sf.write(normalized, samples, rate, subtype='FLOAT')
            acoustic, midi, _ = predict(str(normalized), model_or_model_path=model,
                minimum_frequency=float(librosa.midi_to_hz(45)),
                maximum_frequency=float(librosa.midi_to_hz(96)),
                onset_threshold=.5, frame_threshold=.3, minimum_note_length=90,
                multiple_pitch_bends=False, melodia_trick=False, midi_tempo=120)
        x, (pitches, confidence, rms) = features(samples, rate, acoustic)
        baseline = []
        for part in midi.instruments:
            for note in refine_notes(part.notes, pitches, confidence, 256 / rate, rms):
                baseline.append([note.start, note.end, note.pitch])
        annotations = {}
        for annotator in ['A1', 'A2']:
            annotations[annotator] = np.loadtxt(corpus / 'Annotations' / 'Notes' /
                                               (name + '_notes' + annotator + '.csv'), delimiter=',', ndmin=2)
        np.savez_compressed(target, version=FEATURE_VERSION, x=x,
                            baseline=np.asarray(baseline).reshape(-1, 3), **annotations)
        print(json.dumps({'cached': name, 'seconds': len(samples) / rate, 'frames': len(x)}), flush=True)
    (directory / 'normalized.wav').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
