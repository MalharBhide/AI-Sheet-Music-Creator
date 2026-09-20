"""Repair stale Vienna anchors using audio-only calibration, before prediction."""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


def refine(samples, rate, reference):
    hop = 128
    spectrum = np.abs(librosa.cqt(samples[:15 * rate], sr=rate, hop_length=hop,
                                  fmin=librosa.midi_to_hz(21), n_bins=88))
    log = np.log1p(spectrum * 10)
    flux = np.maximum(0, np.diff(log, axis=1, prepend=log[:, :1]))
    flux /= np.maximum(.001, flux.max(axis=1, keepdims=True))
    onsets = reference[(reference[:, 0] >= .5) & (reference[:, 0] < 14)]
    shifts = np.arange(-.15, .151, .005)
    scores = []
    for shift in shifts:
        scores.append(float(np.mean([flux[int(p) - 21,
            max(0, round((s + shift) * rate / hop) - 2):round((s + shift) * rate / hop) + 3].max()
            for s, _, p in onsets])))
    best = int(np.argmax(scores))
    if best in (0, len(shifts) - 1):
        raise ValueError('Clock correction reaches the predeclared search boundary')
    return float(shifts[best]), scores[best]


def prepare(directory):
    output = directory / 'note-verifier-v3-data'
    if output.exists():
        raise ValueError('Preserve existing dataset manifests')
    manifest = json.loads((directory / 'note-verifier-v2/split.json').read_text())
    anchors = {item['id']: item for item in json.loads((directory / 'vienna-clock-audit.json').read_text())}
    audit = []
    for group, items in manifest['tracks'].items():
        for item in items:
            # Use the exact original accompaniment detector for all candidates.
            item['candidate_decoder'] = 'bounded-accompaniment-v1'
            if item['corpus'] != 'vienna-piano':
                continue
            name = item['id'].removeprefix('vienna-')
            midi = pretty_midi.PrettyMIDI(str(directory / 'vienna/midi' / (name + '.mid')))
            reference = np.asarray([[n.start, n.end, n.pitch] for p in midi.instruments for n in p.notes])
            old = anchors[item['id']]
            # Chopin WAVs start at the music; their 2001 FirstOnsets files refer
            # to earlier untrimmed recordings. The other two works retain silence.
            initial = (-float(reference[:, 0].min()) if name.startswith('Chopin_') else old['shift_seconds'])
            reference[:, :2] += initial
            samples, rate = sf.read(directory / item['audio'], dtype='float32')
            delta, score = refine(samples, rate, reference)
            reference[:, :2] += delta
            keep = (reference[:, 0] >= 15.25) & (reference[:, 0] < 29.75)
            reference = reference[keep]
            reference[:, 1] = np.minimum(reference[:, 1], 30)
            item['reference'] = reference.tolist()
            item['evaluation_window'] = [15.25, 29.75]
            audit.append({'id': item['id'], 'group': group, 'old_shift': old['shift_seconds'],
                          'initial_shift': initial, 'spectral_correction': delta, 'score': score})
    # Keep relative source audio usable from the experiment subdirectory.
    for items in manifest['tracks'].values():
        for item in items:
            item['audio'] = '../' + item['audio']
    manifest['clock_version'] = 2
    manifest['evaluation_status'] = 'All original performers are consumed regression sets; no fresh-test claim'
    manifest['clock_alignment'] = 'Correct stale trimmed Chopin anchors; refine all Vienna clocks with CQT in 0–15s, supervise/evaluate only 15.25–29.75s'
    manifest['release_gates'] = 'For every regression recording: F1 nondecreasing and recall loss <= 0.02 against original detector; corpus precision/F1 nondecreasing; compare with deployed V2; no pitch/timing/velocity changes to retained candidates.'
    output.mkdir()
    (output / 'note-verifier-split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'clock-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'clocks': len(audit), 'manifest_sha256': hashlib.sha256((output / 'note-verifier-split.json').read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
