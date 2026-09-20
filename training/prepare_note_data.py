"""Acquire licensed note-verification data; retain provenance and checksums."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

FILES = [
    ('guitarset-annotations.zip', 'https://zenodo.org/records/3371780/files/annotation.zip?download=1',
     'b39b78e63d3446f2e54ddb7a54df9b10', 'guitarset/annotations'),
    ('guitarset-microphone.zip', 'https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1',
     '275966d6610ac34999b58426beb119c3', 'guitarset/audio'),
    ('oxford-miditest.zip', 'https://www.robots.ox.ac.uk/~vgg/research/sighttosound/resources/MIDItest.zip',
     '10045b377d12499fc2f167338b756401', 'oxford-miditest'),
]


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, url, expected, folder in FILES:
        path = directory / name
        if not path.exists():
            partial = path.with_suffix('.partial')
            with urlopen(url, timeout=60) as response, partial.open('wb') as target:
                size = 0
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    size += len(chunk)
                    if size % (50 * 1024 * 1024) < len(chunk):
                        print(f'{name}: {size // (1024 * 1024)} MiB', flush=True)
            partial.replace(path)
        md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                md5.update(chunk)
                sha.update(chunk)
        if expected and md5.hexdigest() != expected:
            raise ValueError(f'Pinned archive checksum mismatch: {name}')
        destination = (directory / folder).resolve()
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not (destination / item.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe archive path')
            archive.extractall(destination)
        provenance.append({'archive': name, 'url': url, 'md5': md5.hexdigest(),
                           'sha256': sha.hexdigest(), 'bytes': path.stat().st_size,
                           'license': 'CC-BY-4.0'})
        print(json.dumps(provenance[-1]), flush=True)
    (directory / 'note-data-source.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
