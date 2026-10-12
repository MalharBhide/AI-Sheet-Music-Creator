"""Original harmonic/percussive augmentation with native, frozen V12 decisions."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services.bass_verifier import BassVerifier
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from bass_training_data import eligible, identity
from current_bass_v12_baseline import VERSION, hashes
from prepare_bass_positive_data import RATE, original
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode

SEEDS = tuple(range(262401, 262497))
TRAIN_COUNT = 72


def texture(seed):
    """Keep exact played key labels while varying their acoustics and leakage."""
    rng = np.random.default_rng(seed + 817)
    samples, labels = original(seed)
    samples *= .35
    for start, end, pitch in labels:
        first, stop = round(start * RATE), round(end * RATE)
        t = np.arange(stop - first) / RATE
        hz = pretty_midi.note_number_to_hz(pitch)
        coefficients = rng.uniform(.02, .12, 6)
        coefficients[0] = rng.uniform(.015, .1)
        coefficients[int(rng.integers(1, 4))] *= rng.uniform(2., 4.)
        tone = sum(c * np.sin(2 * np.pi * hz * (index + 1) * t + rng.uniform(-.3, .3))
                   for index, c in enumerate(coefficients))
        rise, decay, sustain = rng.uniform(.003, .03), rng.uniform(.18, 3.), rng.uniform(.06, .35)
        envelope = np.minimum(1., t / rise) * np.minimum(1., np.maximum(0., (end - start - t) / .03))
        envelope *= sustain + (1 - sustain) * np.exp(-t / decay)
        samples[first:stop] += (tone * envelope).astype(np.float32)
    # Drum leakage is unpitched accompaniment, not an additional bass key.
    # Labels still include every deliberately played octave from `original`.
    for start in rng.uniform(2.5, 27.5, int(rng.integers(10, 29))):
        first, stop = round(start * RATE), round((start + .4) * RATE)
        t = np.arange(stop - first) / RATE
        low, high, tau = rng.uniform(35., 60.), rng.uniform(95., 165.), rng.uniform(.018, .05)
        phase = 2 * np.pi * (low * t + (high - low) * tau * (1 - np.exp(-t / tau)))
        pulse = (np.sin(phase) + .08 * rng.normal(size=len(t))) * np.exp(-t / rng.uniform(.045, .12))
        pulse *= np.minimum(1., t / .003) * rng.uniform(.015, .075)
        samples[first:stop] += pulse.astype(np.float32)
    samples = np.tanh(samples * rng.uniform(.7, 2.5)).astype(np.float32)
    if not np.isfinite(samples).all() or samples.shape != (30 * RATE,):
        raise ValueError('Invalid original texture waveform')
    return samples, labels


def prepare(parent, consumed, output):
    output.mkdir(exist_ok=False)
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    parent_plan = json.loads((parent / 'plan.json').read_text())
    fresh_manifest = json.loads((consumed / 'manifest.json').read_text())
    if (parent_manifest['plan_sha256'] != digest(parent / 'plan.json')
            or parent_plan['baseline_hashes'] != hashes() or len(parent_manifest['items']) != 705
            or not fresh_manifest['now_consumed_regression'] or len(fresh_manifest['items']) != 32
            or fresh_manifest['plan_sha256'] != digest(consumed / 'plan.json')):
        raise ValueError('Changed current baseline or consumed corpus')
    seen = {r['source_group'] for r in parent_manifest['items'] + fresh_manifest['items']}
    if seen & {'original-bass-seed-' + str(seed) for seed in SEEDS}:
        raise ValueError('Texture sources overlap fitting or consumed regression')
    root = Path(__file__).resolve().parents[1]
    names = ('training/prepare_bass_texture_v12.py', 'training/prepare_bass_positive_data.py',
             'training/prepare_slakh_bass.py', 'backend/app/services/piano_transcription.py',
             'backend/app/services/note_evidence.py')
    plan = {'version': VERSION, 'baseline_hashes': hashes(), 'seeds': list(SEEDS), 'train_count': TRAIN_COUNT,
            'rights': 'Project-authored procedural signals and note sequences',
            'parent_cache_root': str(parent), 'parent_cache_manifest_sha256': digest(parent / 'manifest.json'),
            'parent_cache_plan_sha256': digest(parent / 'plan.json'),
            'consumed_first_pass_root': str(consumed), 'consumed_first_pass_manifest_sha256': digest(consumed / 'manifest.json'),
            'code_sha256': {name: digest(root / name) for name in names},
            'mp3_bitrate_kbps_cycle': [48, 128, 192], 'no_user_audio_or_scores': True,
            'scope': '96 original texture seeds (72 train/24 validation); 705 sealed prior recordings plus 32 newly consumed regressions remain in their partitions. Native V12 baseline only; no fresh-song or deployment claim.'}
    preserve(output / 'plan.json', plan)
    engine, baseline, captured, records = _GeneralEngine('balanced'), BassVerifier(), {}, []
    native_predict = engine.predict_function

    def capture(*args, **kwargs):
        result = native_predict(*args, **kwargs)
        captured['acoustic'] = result[0]
        return result

    engine.predict_function = capture
    for index, seed in enumerate(SEEDS):
        name = 'bass-texture-v1-' + str(seed)
        work = output / name
        work.mkdir()
        samples, labels = texture(seed)
        _, clock = encode(samples, RATE, work, plan['mp3_bitrate_kbps_cycle'][index % 3])
        audio = work / 'decoded.wav'
        samples, rate = sf.read(audio, dtype='float32')
        if rate != RATE or samples.ndim != 1 or len(samples) != RATE * 30:
            raise ValueError('Changed texture codec clock')
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
            midi = engine.predict(audio, 'bass', 90.)
            notes = [n for part in midi.instruments for n in part.notes]
            x = note_features(samples, rate, captured['acoustic'], notes, include_context=True)
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
        p, g = [m.probability(x[:, :m.feature_count]) for m in baseline.models]
        record = {'id': name, 'group': 'train' if index < TRAIN_COUNT else 'validation',
                  'source_group': 'original-bass-seed-' + str(seed), 'corpus': 'original-bass-texture',
                  'audio': str(audio), 'audio_sha256': digest(audio), 'reference': labels.tolist(),
                  'pitch_reference': labels.tolist(), 'seconds': 29.5, 'duration': 30.,
                  'retained_v11': len(events), 'retained_v12': len(events), 'codec_clock': clock}
        # retained_v11 is unknown for this native route; do not mislabel it as
        # a measured earlier-stage count in the audit.
        record.pop('retained_v11')
        path = work / 'features.npz'
        np.savez_compressed(path, base_x=x, events=events, eligible=eligible(events, 30.), context=p, guardian=g,
                            identity=identity(record), plan_sha256=digest(output / 'plan.json'))
        record['cache_sha256'] = digest(path)
        records.append(record)
        print(json.dumps({'cached': len(records), 'id': name, 'group': record['group'], 'v12_retained': len(events)}), flush=True)
    if plan['code_sha256'] != {name: digest(root / name) for name in names} or plan['baseline_hashes'] != hashes():
        raise ValueError('Texture producer changed during native inference')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_user_audio_or_scores': True, 'no_test_inference': True})
    print(json.dumps({'completed': len(records), 'baseline_metadata': engine.bass_verification}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('consumed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.consumed, args.output)
