"""Cache supervised note-verification examples with a fixed performer split."""

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from app.services.note_evidence import FEATURE_VERSION

# Exclude publisher-documented annotation errors before inspecting predictions.
EXCLUDED = ['04_BN3-154-E_comp', '04_Jazz1-200-B_comp', '02_Funk2-119-G_comp']


def guitar_reference(path):
    jams = json.loads(path.read_text())
    notes = [[n['time'], n['time'] + n['duration'], float(n['value'])]
             for annotation in jams['annotations'] if annotation['namespace'] == 'note_midi'
             for n in annotation['data'] if 0 <= n['time'] < 30 and n['duration'] > 0]
    return [[s, min(e, 30), p] for s, e, p in sorted(notes)]


def make_manifest(directory):
    tracks = {'train': [], 'validation': [], 'test': []}
    for annotation in sorted((directory / 'guitarset/annotations').rglob('*.jams')):
        name = annotation.stem
        if name in EXCLUDED:
            continue
        player = name[:2]
        group = 'test' if player == '05' else 'validation' if player == '04' else 'train'
        audio = list((directory / 'guitarset/audio').rglob(name + '_mic.wav'))
        if len(audio) != 1:
            raise ValueError(f'Missing or ambiguous audio: {name}')
        tracks[group].append({'id': name, 'corpus': 'guitarset', 'player': player,
                              'audio': str(audio[0].relative_to(directory)),
                              'reference': guitar_reference(annotation)})
    if [len(tracks[k]) for k in tracks] != [239, 58, 60]:
        raise ValueError('Unexpected GuitarSet manifest')
    for index in range(40):
        name = f'piano-{index:02}'
        reference = json.loads((directory / 'verifier-piano' / (name + '.json')).read_text())
        tracks['train' if index < 32 else 'validation'].append({
            'id': name, 'corpus': 'original-piano', 'audio': f'verifier-piano/{name}.wav',
            'reference': [[n['start'], n['end'], n['pitch']] for n in reference['notes']]})
    return {'version': FEATURE_VERSION, 'guitarset_record': '3371780', 'tracks': tracks,
            'excluded_annotation_errors': EXCLUDED, 'crop': 'First 30 seconds of each recording',
            'split': 'Players 00–03 train, 04 validation, 05 test; original piano seeds 00–31 train, 32–39 validation',
            'independence': 'Performer-disjoint for this verifier; shared progressions, and frozen Basic Pitch pretraining overlap not ruled out',
            'extra_test': 'All eight Oxford MIDItest recordings, untouched until freeze',
            'release_gates': 'Guitar test false positives decrease at least 10%, precision improves, recall drop at most 0.02, F1 nondecreasing. Oxford piano and separated-guitar stress: recall drop at most 0.02 and F1 nondecreasing.'}


def targets(reference, events):
    import librosa
    import mir_eval

    matched = mir_eval.transcription.match_notes(reference[:, :2], librosa.midi_to_hz(reference[:, 2]),
        events[:, :2], librosa.midi_to_hz(events[:, 2]), onset_tolerance=.05, offset_ratio=None)
    y = np.zeros(len(events), dtype=np.float32)
    y[[j for _, j in matched]] = 1
    mask = np.ones(len(events), dtype=bool)
    # A late but correctly pitched detection is an ambiguous training negative.
    # Keep it in evaluation, but do not teach the verifier to erase timing errors.
    for index, (start, end, pitch, _) in enumerate(events):
        if y[index]:
            continue
        same = np.abs(reference[:, 2] - pitch) <= .5
        near = np.abs(reference[:, 0] - start) <= .15
        overlap = np.maximum(0, np.minimum(reference[:, 1], end) - np.maximum(reference[:, 0], start))
        if np.any(same & (near | (overlap >= .5 * (end - start)))):
            mask[index] = False
    return y, mask


_MODEL = None


def cache_one(task):
    import librosa
    import soundfile as sf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    from app.services.note_evidence import note_features

    global _MODEL
    directory, item = task
    directory = Path(directory)
    cache = directory / 'note-verifier-features'
    path = cache / (item['id'] + '.npz')
    digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path) as saved:
            if str(saved['version']) == FEATURE_VERSION and str(saved['identity']) == digest:
                return {'cached': item['id'], 'reused': True}
    if _MODEL is None:
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    samples, rate = librosa.load(directory / item['audio'], sr=22050, mono=True, duration=30)
    audio = cache / (item['id'] + '.wav')
    sf.write(audio, samples, rate, subtype='FLOAT')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        acoustic, midi, _ = predict(str(audio), model_or_model_path=_MODEL,
            onset_threshold=.5, frame_threshold=.3, minimum_note_length=90,
            multiple_pitch_bends=False, melodia_trick=False)
        if item.get('candidate_decoder') == 'bounded-accompaniment-v1':
            from app.services.accompaniment_candidates import decode_candidates

            midi = decode_candidates(acoustic)
        notes = [n for part in midi.instruments for n in part.notes]
        x = note_features(samples, rate, acoustic, notes)
    audio.unlink()
    events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes]).reshape(-1, 4)
    reference = np.asarray(item['reference'], dtype=float).reshape(-1, 3)
    if 'evaluation_window' in item:
        start, stop = item['evaluation_window']
        keep = (events[:, 0] >= start) & (events[:, 0] < stop)
        events, x = events[keep], x[keep]
    y, mask = targets(reference, events)
    np.savez_compressed(path, x=x, events=events, reference=reference, y=y, mask=mask,
                        seconds=(item['evaluation_window'][1] - item['evaluation_window'][0]) if 'evaluation_window' in item else len(samples) / rate, version=FEATURE_VERSION, identity=digest)
    # Do not expose test label counts while preparing features.
    return {'cached': item['id'], 'notes': len(events)}


def run(directory, workers):
    path = directory / 'note-verifier-split.json'
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = make_manifest(directory)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'counts': {k: len(v) for k, v in manifest['tracks'].items()}}), flush=True)
    (directory / 'note-verifier-features').mkdir(exist_ok=True)
    tasks = [(str(directory), item) for entries in manifest['tracks'].values() for item in entries]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = [pool.submit(cache_one, task) for task in tasks]
        for index, future in enumerate(as_completed(pending), 1):
            print(json.dumps({'completed': index, 'total': len(tasks), **future.result()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.directory, args.workers)
