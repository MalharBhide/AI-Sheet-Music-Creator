"""Attach the engraver's real segment positions to the score's playback data."""
import json
import logging
import math
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from app.config import Settings

logger = logging.getLogger(__name__)


def read_positions(source: Path, pages: list[Path]) -> list[dict]:
    sizes = []
    for page in pages:
        root = ET.parse(page).getroot()
        left, top, width, height = map(float, root.attrib['viewBox'].split())
        if left != 0 or top != 0 or width <= 0 or height <= 0:
            raise ValueError('Unsupported score page coordinates')
        sizes.append((width, height))
    root = ET.parse(source).getroot()
    if root.tag != 'score':
        raise ValueError('Invalid segment position export')
    elements = {item.attrib['id']: item.attrib for item in root.findall('./elements/element')}
    positions = []
    # MuseScore 3 .spos units at an explicit 300 DPI are 10 times SVG units:
    # 12 * export DPI / internal DPI (360). Segment widths can be zero in 3.2;
    # use its reliable onset x and full system y/height, never those widths.
    for event in root.findall('./events/event'):
        element = elements[event.attrib['elid']]
        page = int(element['page'])
        if not 0 <= page < len(sizes):
            raise ValueError('Segment refers to a missing page')
        width, height = sizes[page]
        position = {'time': float(event.attrib['position']) / 1000, 'page': page,
                    'x': float(element['x']) / (10 * width),
                    'y': float(element['y']) / (10 * height),
                    'height': float(element['sy']) / (10 * height)}
        if (not all(math.isfinite(value) for value in position.values())
                or position['time'] < 0 or not 0 <= position['x'] <= 1
                or not 0 <= position['y'] < 1 or not 0 < position['height'] <= 1
                or position['y'] + position['height'] > 1.01):
            raise ValueError('Invalid segment coordinates')
        positions.append(position)
    if not positions or any(a['time'] > b['time'] for a, b in zip(positions, positions[1:], strict=False)):
        raise ValueError('Missing or unordered segment timing')
    return positions


def attach_score_positions(xml_path: Path, directory: Path, settings: Settings) -> bool:
    """Optional preview enhancement; a layout failure never loses the music."""
    playback_path = directory / 'playback.json'
    pages = sorted(directory.glob('page-*.svg'), key=lambda p: int(p.stem.split('-')[-1]))
    if not playback_path.exists() or not pages:
        return False
    source = directory / 'score.spos'
    temporary = directory / 'playback-positions.tmp'
    try:
        renderer = settings.renderer()
        if not renderer:
            return False
        runtime = directory / 'qt-runtime'
        runtime.mkdir(mode=0o700, exist_ok=True)
        subprocess.run([renderer, '-r', '300', '-o', str(source), str(xml_path)],
                       env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
                            'XDG_RUNTIME_DIR': str(runtime)},
                       capture_output=True, check=True,
                       timeout=settings.render_timeout_seconds or None)
        positions = read_positions(source, pages)
        playback = json.loads(playback_path.read_text())
        if positions[-1]['time'] > playback['duration'] + .1:
            raise ValueError('Position map extends past score playback')
        playback['positions'] = positions
        temporary.write_text(json.dumps(playback, separators=(',', ':'), allow_nan=False))
        temporary.replace(playback_path)
        return True
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.SubprocessError):
        logger.warning('Score following unavailable; retaining playback and downloads', exc_info=True)
        return False
    finally:
        source.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    from app.services.job_store import JobStore

    settings = Settings()
    store = JobStore(settings.database_path)
    for directory in settings.jobs_dir.iterdir():
        job = store.get(directory.name)
        if job and job['status'] == 'completed':
            output = directory / 'outputs'
            playback_path = output / 'playback.json'
            if playback_path.is_file() and not json.loads(playback_path.read_text()).get('positions'):
                success = attach_score_positions(output / 'score.musicxml', output, settings)
                print(f'{directory.name}: {"following added" if success else "following unavailable"}')
