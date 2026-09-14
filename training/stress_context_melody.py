"""Frozen V2/V1 comparison after Demucs on eight preselected unseen voices."""

import argparse
import contextlib
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model
from basic_pitch.inference import predict as acoustic_predict
from melody_candidate import ContextMelodyDecoder
from stress_melody import backing
from train_context_melody import baseline_model
from train_melody import evaluate, predict

from app.services.melody_decoder import decode, features
from app.services.source_separation import StemSeparator


def run(directory):
    torch.set_num_threads(2)
    run_path = directory / 'context-melody-v2'
    target = run_path / 'separated-test.json'
    if not (run_path / 'test.json').exists() or target.exists():
        raise ValueError('Run the clean test first; separated test can be consumed only once')
    selection = json.loads((run_path / 'selection.json').read_text())
    digest = hashlib.sha256((run_path / 'candidate.pt').read_bytes()).hexdigest()
    if digest != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    checkpoint = torch.load(run_path / 'candidate.pt', weights_only=True, map_location='cpu')
    candidate = ContextMelodyDecoder()
    candidate.load_state_dict(checkpoint['state_dict'])
    baseline, released = baseline_model()
    manifest = json.loads((directory / 'vocalset-split.json').read_text())
    # Fixed before looking at either system's results: first two clips per
    # singer in the seed-frozen manifest, no filtering by accuracy.
    counts, selected = Counter(), []
    for item in manifest['tracks']['test']:
        if counts[item['singer']] < 2:
            selected.append(item)
            counts[item['singer']] += 1
    separator, acoustic = StemSeparator(), Model(ICASSP_2022_MODEL_PATH)
    dataset = []
    for index, item in enumerate(selected):
        work = directory / 'context-stress' / item['id']
        work.mkdir(parents=True, exist_ok=True)
        voice, rate = librosa.load(directory / 'annotated-vocalset/VocalSet' / item['audio'],
                                   sr=22050, duration=30)
        support = backing(len(voice), rate, 260914 + index)
        support *= np.sqrt(np.mean(voice ** 2)) / max(1e-8, np.sqrt(np.mean(support ** 2))) * 10 ** (-3 / 20)
        mixture = voice + support
        mixture *= .95 / max(.95, float(np.max(np.abs(mixture))))
        sf.write(work / 'mixture.wav', mixture, rate, subtype='FLOAT')
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            stems = separator.separate(work / 'mixture.wav', work)
            path = work / 'vocals.wav'
            if 'vocals' not in stems:
                sf.write(path, np.zeros_like(voice), rate, subtype='FLOAT')
            arrays, _, _ = acoustic_predict(str(path), model_or_model_path=acoustic,
                onset_threshold=.5, frame_threshold=.3, minimum_note_length=90,
                multiple_pitch_bends=False, melodia_trick=False)
            samples, rate = sf.read(path, dtype='float32')
            x, _ = features(samples, rate, arrays)
        with torch.inference_mode():
            logits, attacks = baseline(torch.from_numpy(x)[None])
        events = decode(logits[0].numpy(), torch.sigmoid(attacks[0]).numpy(), **released['decoder'])
        reference = np.asarray(item['notes'])
        dataset.append({'id': item['id'], 'x': x, 'baseline': events, 'A1': reference})
        np.savez_compressed(work / 'features.npz', x=x, baseline=events, reference=reference)
        print(json.dumps({'separated': item['id']}), flush=True)
    report = evaluate(dataset, predict(candidate, dataset), checkpoint['decoder'])
    report.update({'checkpoint_sha256': digest, 'backing_level_db': -3,
                   'selection': 'First two recordings per test singer in frozen manifest',
                   'description': 'Real unseen singing with procedural backing passed through Demucs; not commercial mixed-song accuracy'})
    current, old = report['aggregate']['trained'], report['aggregate']['baseline']
    report['passes'] = (current['notes']['f1'] > old['notes']['f1']
                        and current['notes_with_offsets']['f1'] > old['notes_with_offsets']['f1']
                        and current['voiced_pitch_recall'] >= old['voiced_pitch_recall']
                        and current['false_voicing'] <= old['false_voicing'] + .02
                        and current['false_silence'] <= old['false_silence'] + .02)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'aggregate': report['aggregate'], 'passes': report['passes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
