"""Select singer-disjoint VocalSet examples and cache frozen acoustic features.

The nominal score can be in a different key/octave than the performance. Use
one per-recording transposition inferred from the dataset's reviewed pitch
annotations, rejecting inconsistent annotation/score pairs before splitting.
Never infer reference labels from this application's predictions.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SEED = 260913


def annotation_clock_scale(audio_duration, annotated_duration):
    """Reconcile only exact clock-rate mismatches supported by file metadata."""
    if audio_duration <= 0 or annotated_duration <= 0:
        raise ValueError('Invalid audio or annotation duration')
    ratio = audio_duration / annotated_duration
    for factor in (1., .5, 2.):
        if abs(ratio / factor - 1) <= .02:
            return factor
    raise ValueError('Audio and annotation durations disagree; review this recording')


def reconcile_clocks(manifest, directory):
    import soundfile as sf

    corpus = directory / 'annotated-vocalset' / 'VocalSet'
    rows = defaultdict(list)
    with (corpus / 'annotations' / 'extended 2' / 'all files.csv').open() as source:
        for row in csv.DictReader(source, skipinitialspace=True):
            rows[row['File Name']].append(row)
    for group in manifest['tracks'].values():
        for item in group:
            entries = rows[item['id']]
            scale = annotation_clock_scale(sf.info(str(corpus / item['audio'])).duration,
                                           float(entries[0]['Total Duration']))
            # Re-read all rows before cropping: scaling already-cropped labels
            # would accidentally erase the second half of a long recording.
            notes = []
            for row in entries:
                if row['Type'] != 'Sound':
                    continue
                start, end = float(row['Start time']) * scale, float(row['End time']) * scale
                pitch = float(row['Ground Truth MIDI code']) + item['transpose']
                if 0 <= start < 30:
                    notes.append([start, 440 * 2 ** ((pitch - 69) / 12), min(end, 30) - start])
            item['notes'] = notes
            item['annotation_clock_scale'] = scale
    manifest['annotation_clock_version'] = 1
    return manifest


def make_manifest(directory):
    import numpy as np

    corpus = directory / 'annotated-vocalset' / 'VocalSet'
    audio = defaultdict(list)
    for path in (corpus / 'FULL').rglob('*.wav'):
        audio[path.stem].append(path)
    rows = defaultdict(list)
    with (corpus / 'annotations' / 'extended 2' / 'all files.csv').open() as source:
        for row in csv.DictReader(source, skipinitialspace=True):
            rows[row['File Name']].append(row)
    eligible, exclusions = [], Counter()
    for name, entries in sorted(rows.items()):
        if len(audio[name]) != 1:
            exclusions['missing_or_ambiguous_audio'] += 1
            continue
        if 'spoken' in name:
            exclusions['speech_without_defined_melody'] += 1
            continue
        sounding = [r for r in entries if r['Type'] == 'Sound']
        try:
            estimates = np.array([float(r['Estimated MIDI code']) for r in sounding])
            nominal = np.array([float(r['Ground Truth MIDI code']) for r in sounding])
            starts = np.array([float(r['Start time']) for r in sounding])
            ends = np.array([float(r['End time']) for r in sounding])
        except (ValueError, TypeError):
            exclusions['missing_pitch_or_timing_labels'] += 1
            continue
        if not len(sounding) or not np.isfinite(np.r_[estimates, nominal, starts, ends]).all():
            exclusions['invalid_annotations'] += 1
            continue
        transpose = int(round(float(np.median(estimates - nominal))))
        pitches = nominal + transpose
        if (np.mean(np.abs(estimates - pitches) <= .8) < .9
                or np.any(ends <= starts) or np.any(pitches < 21) or np.any(pitches > 108)):
            exclusions['inconsistent_performed_pitch_or_timing'] += 1
            continue
        # Crop every selected recording identically, to one production-sized
        # window. Keep real rests; do not choose excerpts by model success.
        notes = [[float(s), float(440 * 2 ** ((p - 69) / 12)), float(min(e, 30) - s)]
                 for s, e, p in zip(starts, ends, pitches, strict=True) if 0 <= s < 30]
        if not notes:
            exclusions['no_notes_in_window'] += 1
            continue
        entry = entries[0]
        eligible.append({'id': name, 'singer': entry['Singer Name'],
                         'category': entry['Technique'] + '/' + entry['Music Type'],
                         'audio': str(audio[name][0].relative_to(corpus)),
                         'transpose': transpose, 'notes': notes,
                         'label_pitch_agreement': float(np.mean(np.abs(estimates - pitches) <= .8))})
    singers = sorted({r['singer'] for r in eligible})
    rng = random.Random(SEED)
    split_singers = {'train': [], 'validation': [], 'test': []}
    for prefix in ['f', 'm']:
        group = [s for s in singers if s.startswith(prefix)]
        rng.shuffle(group)
        split_singers['test'].extend(group[:2])
        split_singers['validation'].extend(group[2:4])
        split_singers['train'].extend(group[4:])
    selected = {key: [] for key in split_singers}
    for split, group in split_singers.items():
        for singer in group:
            categories = defaultdict(list)
            for entry in eligible:
                if entry['singer'] == singer:
                    categories[entry['category']].append(entry)
            for examples in categories.values():
                rng.shuffle(examples)
            keys = sorted(categories)
            rng.shuffle(keys)
            count, limit = 0, 24 if split == 'train' else 12
            while count < limit and any(categories.values()):
                for key in keys:
                    if categories[key] and count < limit:
                        selected[split].append(categories[key].pop())
                        count += 1
    manifest = {'seed': SEED, 'corpus_record': '10200775', 'variant': 'extended 2',
            'singers': split_singers, 'tracks': selected, 'exclusions': dict(exclusions),
            'eligible_files': len(eligible), 'window_seconds': 30,
            'sampling': 'Category round robin per singer: 24 train, 12 validation/test',
            'scope': 'Singer-disjoint; shared exercises, not composition-disjoint'}
    return reconcile_clocks(manifest, directory)


_MODEL = None


def cache_one(task):
    import contextlib
    import os

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    from app.services.melody_decoder import FEATURE_VERSION, MelodyDecoder, decode, features

    global _MODEL
    directory, item = task
    directory = Path(directory)
    cache = directory / 'vocalset-features'
    target = cache / (item['id'] + '.npz')
    label_hash = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    if target.exists():
        with np.load(target) as saved:
            if str(saved['label_hash']) == label_hash and str(saved['version']) == FEATURE_VERSION:
                return {'cached': item['id'], 'reused': True}
    torch.set_num_threads(2)
    if _MODEL is None:
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    path = directory / 'annotated-vocalset' / 'VocalSet' / item['audio']
    samples, rate = librosa.load(path, sr=22050, mono=True, duration=30)
    normalized = cache / (item['id'] + '.wav')
    sf.write(normalized, samples, rate, subtype='FLOAT')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        acoustic, _, _ = predict(str(normalized), model_or_model_path=_MODEL,
            onset_threshold=.5, frame_threshold=.3, minimum_note_length=90,
            multiple_pitch_bends=False, melodia_trick=False)
    normalized.unlink()
    x, _ = features(samples, rate, acoustic)
    # Freeze the released model as baseline before any new fitting or selection.
    asset = Path(__file__).resolve().parents[1] / 'backend/app/assets/vocal-melody-v1.pt'
    checkpoint = torch.load(asset, weights_only=True, map_location='cpu')
    model = MelodyDecoder().eval()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.inference_mode():
        logits, attacks = model(torch.from_numpy(x)[None])
    baseline = decode(logits[0].numpy(), torch.sigmoid(attacks[0]).numpy(), **checkpoint['decoder'])
    np.savez_compressed(target, x=x, baseline=baseline, reference=np.asarray(item['notes']),
                        version=FEATURE_VERSION, label_hash=label_hash)
    return {'cached': item['id'], 'seconds': len(samples) / rate}


def run(directory, workers):
    manifest_path = directory / 'vocalset-split.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('annotation_clock_version') != 1:
            raise ValueError('Old annotation clocks: run repair_vocalset_clocks.py before using this cache')
    else:
        manifest = make_manifest(directory)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'singers': manifest['singers'], 'eligible': manifest['eligible_files'],
                      'selected': {k: len(v) for k, v in manifest['tracks'].items()},
                      'exclusions': manifest['exclusions']}), flush=True)
    (directory / 'vocalset-features').mkdir(exist_ok=True)
    tasks = [(str(directory), item) for group in manifest['tracks'].values() for item in group]
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
