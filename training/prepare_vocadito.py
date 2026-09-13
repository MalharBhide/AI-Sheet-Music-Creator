"""Download the pinned, CC BY 4.0 Vocadito corpus and verify its checksum."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = 'https://zenodo.org/records/5578807/files/vocadito.zip?download=1'
MD5 = 'dea40fd18f14d899643c4ba221b33a46'


def prepare(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'vocadito.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        try:
            with urlopen(URL, timeout=60) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.md5(usedforsecurity=False)
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        raise ValueError('Vocadito archive checksum does not match the publisher')
    destination = (directory / 'vocadito').resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive member escapes the dataset directory')
        source.extractall(destination)
    (directory / 'dataset-source.json').write_text(json.dumps({
        'dataset': 'Vocadito', 'version': 'Zenodo 5578807', 'url': URL,
        'md5': MD5, 'license': 'CC-BY-4.0',
        'authors': 'Rachel Bittner, Katherine Pasalo, Juan Jose Bosch, Gabriel Meseguer Brocal, David Rubinstein',
        'source': 'https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/vocadito.html',
    }, indent=2) + '\n')
    print('Verified and extracted', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
