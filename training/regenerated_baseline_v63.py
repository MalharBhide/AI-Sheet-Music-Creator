"""Rebuild fitting observations from pinned audio, without forging legacy caches.

The original readers and archived cache hashes remain unchanged. This is a new,
explicitly identified feature recipe using the same 908 fitting recordings and
labels, through the unchanged native V32 decoder. It cannot release weights.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.piano_transcription import _GeneralEngine
from prepare_left_support_v41 import (
    KEYS, RATE, coarse_sequences,
    context_features, eligible, hashes, neighbor_features, note_features,
    recurrence, recurrence_observe, release_sequences,
)
from attack_local_features import observe as acoustic_observe, sequences as attack_sequences
from periodicity_bass_data import validate
from prepare_robust_training_stems import digest, preserve

VERSION = 'native-v32-regenerated-fitting-baseline-v63-v1'
COUNTS = {'train': 702, 'validation': 206}
WITNESSES = {
    'bass-v25-baseline-v1': '9f6e55b9b7588e5bace6b45d949d8b9b5f6fa323f2ba69f0eb044df055dd1555',
    'nsynth-bass-v25-v1': '4691accb4a2730e47ad83c7671744c941551d63d2faf429f127057b099fc7dc9',
    'groove-bass-v29-v1': 'fc887f7de53e98dfb6aa8eed83b9fc4b5f13d9428534576da51bf15f9e9809a6',
}
FIELDS = ('id', 'group', 'source_group', 'corpus', 'audio', 'audio_sha256',
          'duration', 'seconds', 'reference', 'pitch_reference')


def contracts():
    from prepare_left_support_v41 import contracts as physical_contracts
    return {**physical_contracts(), 'regenerated_baseline_v63.py': digest(Path(__file__))}


def validate_registry(rows, counts=COUNTS):
    if (len(rows) != sum(counts.values()) or len({r['id'] for r in rows}) != len(rows)
            or dict(Counter(r['group'] for r in rows)) != counts):
        raise ValueError('Incomplete or duplicate fitting source registry')
    partitions = {g: {r['source_group'] for r in rows if r['group'] == g} for g in counts}
    if partitions['train'] & partitions['validation']:
        raise ValueError('Fitting source-group leakage')
    for row in rows:
        path = Path(row['audio'])
        if (Path(row['id']).name != row['id'] or row['id'] in ('.', '..')
                or not path.is_absolute() or not path.is_relative_to('/training')
                or '..' in path.parts or not np.isfinite(row['duration']) or row['duration'] <= 0
                or not np.isfinite(row['seconds']) or not 0 < row['seconds'] <= row['duration']):
            raise ValueError('Invalid fitting source path or clock')
        for key in ('reference', 'pitch_reference'):
            refs = np.asarray(row[key], float).reshape(-1, 3)
            if (not np.isfinite(refs).all() or np.any(refs[:, 0] < 0)
                    or np.any(refs[:, 1] <= refs[:, 0])
                    or np.any(refs[:, 1] > row['duration'] + 1e-6)
                    or np.any((refs[:, 2] < 21) | (refs[:, 2] >= 60))
                    or (row['corpus'] != 'guitarset-bass-register'
                        and np.any(refs[:, 2] != np.floor(refs[:, 2])))):
                raise ValueError('Invalid pinned reference clock or pitch')
    return rows


def registry(production, training, verify_audio=True):
    rows = []
    for name, expected in WITNESSES.items():
        path = production / 'training/results' / name / 'manifest.json'
        if digest(path) != expected:
            raise ValueError('Changed original fitting-label witness')
        manifest = json.loads(path.read_text())
        for original in manifest['items']:
            row = {key: original[key] for key in FIELDS}
            row['source_manifest_sha256'] = expected
            row['source_record_sha256'] = hashlib.sha256(
                json.dumps(original, sort_keys=True).encode()).hexdigest()
            rows.append(row)
    validate_registry(rows)
    for row in rows:
        audio = Path(row['audio'])
        if (audio.is_symlink() or not audio.resolve().is_relative_to(training.resolve())):
            raise ValueError('Fitting audio escapes audited training storage')
        if verify_audio:
            if digest(audio) != row['audio_sha256']:
                raise ValueError('Changed original fitting waveform: ' + row['id'])
            info = sf.info(audio)
            if (info.samplerate != RATE or info.channels != 1
                    or info.frames != round(row['duration'] * RATE)):
                raise ValueError('Changed physical fitting clock')
    return rows


def check_arrays(record, arrays):
    events = arrays['events']
    if (events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > record['duration'] + 1e-6)
            or np.any((events[:, 2] < 21) | (events[:, 2] >= 60))
            or np.any(events[:, 2] != np.floor(events[:, 2]))
            or np.any((events[:, 3] < 1) | (events[:, 3] > 127))):
        raise ValueError('Invalid native candidate intervals')
    for key, shape in (('frames', (40, 9)), ('attack_frames', (61, 18)),
                       ('release_frames', (61, 18))):
        value = arrays[key]
        if (value.shape != (len(events), *shape) or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError('Invalid regenerated physical observations')
    if arrays['eligible'].dtype != np.bool_ or arrays['eligible'].shape != (len(events),):
        raise ValueError('Invalid eligibility mask')
    np.testing.assert_array_equal(arrays['eligible'], eligible(events, record['duration']))
    np.testing.assert_array_equal(arrays['x'], context_features(events, arrays['base_x']))
    np.testing.assert_array_equal(arrays['release_neighbors'], neighbor_features(events, arrays['base_x']))
    return validate({**record, **arrays,
        'reference': np.asarray(record['reference'], float).reshape(-1, 3),
        'pitch_reference': np.asarray(record['pitch_reference'], float).reshape(-1, 3)})


def read_record(folder, record, original, plan_sha):
    if (any(record[k] != v for k, v in original.items())
            or record['plan_sha256'] != plan_sha
            or digest(Path(record['audio'])) != record['audio_sha256']):
        raise ValueError('Changed regenerated source, labels or partition')
    path = folder / 'features' / (record['id'] + '.npz')
    if digest(path) != record['cache_sha256']:
        raise ValueError('Changed regenerated feature cache')
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['plan_sha256']) != plan_sha:
            raise ValueError('Changed regenerated cache identity')
        arrays = {key: saved[key] for key in KEYS}
    if record['notes'] != len(arrays['events']):
        raise ValueError('Changed regenerated candidate count')
    return check_arrays(record, arrays)


def prepare(production, training, output):
    from approved_model_finetune import storage_preflight
    print(json.dumps({'storage': storage_preflight(training, 2_000_000_000,
        audited_links={'left-precision-v43-replay/training/results': production / 'training/results'})}), flush=True)
    rows = registry(production, training)
    plan = {'version': VERSION, 'production_root': str(production), 'training_root': str(training),
        'label_witnesses': WITNESSES, 'groups': COUNTS, 'native_v32': hashes(),
        'code_sha256': contracts(), 'source_registry': rows,
        'regenerated_features_claimed_identical_to_historical_cache': False,
        'native_full_window_dataset_inference': True, 'no_user_audio_or_scores': True,
        'no_consumed_or_reserved_source_read': True, 'new_weights_deployed': False}
    if output.exists():
        if ((output / 'manifest.json').exists()
                or json.loads((output / 'plan.json').read_text()) != plan):
            raise ValueError('Preserve changed or completed regenerated observations')
    else:
        output.mkdir()
        preserve(output / 'plan.json', plan)
    (output / 'features').mkdir(exist_ok=True)
    (output / 'records').mkdir(exist_ok=True)
    engine = _GeneralEngine('balanced')
    native, captured = engine.predict_function, {}
    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured['acoustic'] = result[0]
        return result
    engine.predict_function = capture
    records = []
    for original in rows:
        completion = output / 'records' / (original['id'] + '.json')
        if completion.exists():
            record = json.loads(completion.read_text())
            read_record(output, record, original, digest(output / 'plan.json'))
        else:
            audio = Path(original['audio'])
            if digest(audio) != original['audio_sha256']:
                raise ValueError('Waveform changed during native preparation')
            wave, rate = sf.read(audio, dtype='float32')
            if rate != RATE or not np.isfinite(wave).all():
                raise ValueError('Invalid native input samples')
            midi = engine.predict(audio, 'bass', 120.)
            notes = [n for part in midi.instruments for n in part.notes]
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
            base = note_features(wave, rate, captured['acoustic'], notes, include_context=True)
            observed = acoustic_observe(wave, rate)
            arrays = {'events': events, 'base_x': base, 'x': context_features(events, base),
                'frames': coarse_sequences(observed['cqt'], events),
                'attack_frames': attack_sequences(observed, events),
                'release_frames': release_sequences(observed, events),
                'release_neighbors': neighbor_features(events, base),
                'periodicity': recurrence(recurrence_observe(wave, rate), events),
                'eligible': eligible(events, original['duration'])}
            check_arrays(original, arrays)
            path = output / 'features' / (original['id'] + '.npz')
            np.savez_compressed(path, **arrays, plan_sha256=digest(output / 'plan.json'))
            record = {**original, 'notes': len(events), 'cache_sha256': digest(path),
                      'plan_sha256': digest(output / 'plan.json')}
            preserve(completion, record)
        records.append(record)
        if len(records) % 10 == 0:
            if plan['native_v32'] != hashes() or plan['code_sha256'] != contracts():
                raise ValueError('Native feature producer changed')
        print(json.dumps({'complete': len(records), 'total': 908, 'id': original['id']}), flush=True)
    if plan['native_v32'] != hashes() or plan['code_sha256'] != contracts():
        raise ValueError('Native feature producer changed')
    preserve(output / 'manifest.json', {'version': VERSION, 'items': records,
        'plan_sha256': digest(output / 'plan.json'), 'no_user_audio_or_scores': True,
        'no_consumed_or_reserved_source_read': True, 'new_weights_deployed': False})


def load(folder):
    plan = json.loads((folder / 'plan.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (plan['version'] != VERSION or manifest['version'] != VERSION
            or plan['label_witnesses'] != WITNESSES or plan['groups'] != COUNTS
            or plan['native_v32'] != hashes() or plan['code_sha256'] != contracts()
            or manifest['plan_sha256'] != digest(folder / 'plan.json')
            or plan['regenerated_features_claimed_identical_to_historical_cache']
            or plan['native_full_window_dataset_inference'] is not True
            or any(not doc[key] for doc in (plan, manifest) for key in
                   ('no_user_audio_or_scores', 'no_consumed_or_reserved_source_read'))
            or any(doc['new_weights_deployed'] for doc in (plan, manifest))):
        raise ValueError('Changed regenerated fitting contract')
    originals = registry(Path(plan['production_root']), Path(plan['training_root']))
    if plan['source_registry'] != originals:
        raise ValueError('Changed complete original fitting registry')
    records = manifest['items']
    if (len(records) != 908 or len({r['id'] for r in records}) != 908
            or {r['id'] for r in records} != {r['id'] for r in originals}):
        raise ValueError('Incomplete regenerated fitting observations')
    by_id = {r['id']: r for r in originals}
    return [read_record(folder, r, by_id[r['id']], manifest['plan_sha256']) for r in records]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('production', 'training', 'output'):
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
