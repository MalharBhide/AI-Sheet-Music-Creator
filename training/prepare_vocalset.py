"""Fetch the pinned CC BY 4.0 audio/annotation revision of VocalSet."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = 'https://zenodo.org/records/10200775/files/VocalSet.zip?download=1'
MD5 = '8d39344bbc775aa040840783ae73cfa4'


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'annotated-vocalset.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        with urlopen(URL, timeout=60) as source, temporary.open('wb') as target:
            copied = 0
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
                copied += len(chunk)
                if copied // (100 * 1024 * 1024) != (copied - len(chunk)) // (100 * 1024 * 1024):
                    print(f'Downloaded {copied // (1024 * 1024)} MiB', flush=True)
        temporary.replace(archive)
    digest = hashlib.md5(usedforsecurity=False)
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        raise ValueError('VocalSet archive differs from the publisher checksum')
    destination = (directory / 'annotated-vocalset').resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive entry escapes the dataset directory')
        source.extractall(destination)
    (directory / 'vocalset-source.json').write_text(json.dumps({
        'dataset': 'VocalSet & Annotated VocalSet', 'record': '10200775',
        'url': URL, 'md5': MD5, 'license': 'CC-BY-4.0',
        'revision_author': 'Santiago Donaher',
        'audio_authors': 'Julia Wilkins, Prem Seetharaman, Alison Wahl, Bryan Pardo',
        'annotation_authors': 'Behnam Faghih, Joseph Timoney',
        'upstream_records': ['1193957', '7061507'],
        'labels': 'Semi-automatic pitch and boundary estimates reviewed and corrected by annotators',
    }, indent=2) + '\n')
    print('Verified and extracted', destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
