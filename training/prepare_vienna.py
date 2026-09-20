"""Acquire and align the CC BY 4.0 Vienna 4x22 performance corpus."""

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

AUDIO_URL = 'https://datasets.mdw.ac.at/media/content_files/8dec8d0a-c4da-4a96-b290-a3ba783c03ba.zip'
MIDI_URL = 'https://datasets.mdw.ac.at/media/content_files/26a437a4-6130-4f66-9488-20a0fae91751.zip'
SIZE = 1295130820
CHECKSUMS = {
    AUDIO_URL: '23d31fcb6eb88cb0991376270688c98129b4d2104dbe3b1d9f458d01403b0f9c',
    MIDI_URL: '104adc24bc5bccc133fa174ff0055d88cd7a237005d233df52787bdc15d8e6af',
}


def acquire(directory):
    path = directory / 'vienna-audio.zip'
    if not zipfile.is_zipfile(path):
        chunks = directory / 'vienna-download'
        chunks.mkdir(exist_ok=True)
        length = 32 * 1024 * 1024

        def download(index):
            start, end = index * length, min(SIZE, (index + 1) * length) - 1
            target = chunks / str(index)
            if target.exists() and target.stat().st_size == end - start + 1:
                return target
            with urlopen(Request(AUDIO_URL, headers={'Range': f'bytes={start}-{end}'}), timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Server did not honor the exact requested byte range')
                data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError('Truncated dataset range')
                target.write_bytes(data)
            print(json.dumps({'downloaded_part': index}), flush=True)
            return target

        with ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(download, range((SIZE + length - 1) // length)))
        with path.open('wb') as out:
            for part in parts:
                out.write(part.read_bytes())
    midi = directory / 'vienna-midi.zip'
    if not midi.exists():
        with urlopen(MIDI_URL, timeout=90) as response:
            midi.write_bytes(response.read())
    provenance = []
    for archive, url in ((path, AUDIO_URL), (midi, MIDI_URL)):
        sha = hashlib.sha256()
        with archive.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                sha.update(block)
        if sha.hexdigest() != CHECKSUMS[url]:
            raise ValueError('Pinned Vienna archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            destination = (directory / 'vienna').resolve()
            for entry in z.infolist():
                if not (destination / entry.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe dataset archive path')
            z.extractall(destination)
        provenance.append({'url': url, 'sha256': sha.hexdigest(), 'bytes': archive.stat().st_size})
    (directory / 'vienna-source.json').write_text(json.dumps({'author': 'Werner Goebl',
        'doi': '10.21939/4X22', 'license': 'CC-BY-4.0', 'archives': provenance}, indent=2) + '\n')
    print(json.dumps(provenance), flush=True)




def prepare_split(directory):
    import re

    import librosa
    import numpy as np
    import pretty_midi
    import soundfile as sf
    from repair_verifier_clocks import refine

    manifest = json.loads((directory / 'note-verifier-v1/split.json').read_text())
    manifest['previously_consumed_tests'] = ['GuitarSet player 05', 'Oxford MIDItest; regression only after first candidate']
    manifest['release_gates'] += ' Fresh Vienna players 19–22: recall drop <= 0.02 and F1 nondecreasing.'
    manifest['split'] += '; Vienna players 01–14 train, 15–18 validation, 19–22 test; performer 23 average excluded'
    output = directory / 'vienna-aligned'
    output.mkdir(exist_ok=True)
    audit = []
    for path in sorted((directory / 'vienna/midi').glob('*.mid')):
        match = re.search(r'_p(\d+)', path.stem)
        if match is None or int(match[1]) > 22 or '1st-3rd' in path.stem:
            continue
        player = int(match[1])
        files = list((directory / 'vienna/audio').rglob(path.stem + '.*'))
        if len(files) != 1:
            raise ValueError(f'Audio pairing failed: {path.stem}: {files}')
        samples, rate = librosa.load(files[0], sr=22050, duration=30)
        midi = pretty_midi.PrettyMIDI(str(path))
        reference = np.asarray([[n.start, n.end, n.pitch] for part in midi.instruments for n in part.notes])
        onset_files = list(files[0].parent.glob('*FirstOnsets.txt'))
        if len(onset_files) != 1:
            raise ValueError('Missing publisher clock anchors')
        anchors = {int(line.split()[0]): float(line.split()[1])
                   for line in onset_files[0].read_text().splitlines()
                   if line.strip() and not line.startswith('%')}
        # Chopin audio has been trimmed; publisher anchors refer to untrimmed
        # recordings. Calibrate on 0–15s, then exclude that region from labels.
        initial = (0. if path.stem.startswith('Chopin_') else anchors[player]) - float(reference[:, 0].min())
        reference[:, :2] += initial
        delta, score = refine(samples, rate, reference)
        reference[:, :2] += delta
        record = {'shift_seconds': initial + delta, 'publisher_first_onset': anchors[player],
                  'initial_shift': initial, 'spectral_correction': delta, 'score': score,
                  'source': str(onset_files[0].relative_to(directory))}
        reference = reference[(reference[:, 0] >= 15.25) & (reference[:, 0] < 29.75)]
        reference[:, 1] = np.minimum(reference[:, 1], 30)
        name = 'vienna-' + path.stem
        audio = output / (name + '.wav')
        sf.write(audio, samples, rate, subtype='FLOAT')
        group = 'train' if player <= 14 else 'validation' if player <= 18 else 'test'
        manifest['tracks'][group].append({'id': name, 'corpus': 'vienna-piano', 'player': player,
            'audio': str(audio.relative_to(directory)), 'reference': reference.tolist(),
            'evaluation_window': [15.25, 29.75]})
        audit.append({'id': name, **record})
    if len(audit) != 88:
        raise ValueError(f'Expected 88 real performances, found {len(audit)}')
    manifest['tracks']['validation'].extend(json.loads((directory / 'note-verifier-stems-validation/manifest.json').read_text())['items'])
    manifest['clock_version'] = 2
    manifest['clock_alignment'] = 'Vienna: correct trimmed Chopin clocks; audio-only CQT calibration on 0–15s; labels only 15.25–29.75s. Exclude special 1st-3rd Ballade variants.'
    (directory / 'vienna-clock-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (directory / 'note-verifier-split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'counts': {k: len(v) for k, v in manifest['tracks'].items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--align', action='store_true')
    args = parser.parse_args()
    if args.align:
        prepare_split(args.directory)
    else:
        acquire(args.directory)
