"""Pinned public model assets, downloaded before jobs in the Docker build."""
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from app.models import PipelineError

PIANO_MODEL_URL = (
    'https://zenodo.org/records/4034264/files/'
    'CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1'
)
# Publisher-provided checksum from https://zenodo.org/api/records/4034264.
PIANO_MODEL_MD5 = '22b961b77c1878239fec963362097045'


def ensure_piano_model(destination: Path | None = None) -> Path:
    destination = destination or Path.home() / '.cache/piano-scribe/piano.pth'
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with urlopen(PIANO_MODEL_URL, timeout=60) as response, temporary.open('wb') as target:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != PIANO_MODEL_MD5:
            raise ValueError('Model checksum does not match the publisher')
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise PipelineError('The piano model could not be downloaded. Check the server connection and retry, or configure PIANO_MODEL_PATH.') from exc
    return destination


if __name__ == '__main__':
    from demucs.pretrained import get_model

    destination = os.environ.get('PIANO_MODEL_PATH')
    ensure_piano_model(Path(destination) if destination else None)
    get_model('htdemucs')
