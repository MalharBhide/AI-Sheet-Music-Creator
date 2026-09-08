import subprocess
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.models import PipelineError
from app.services.score_render import render_score


def test_renderer_collects_every_svg_page_in_numeric_order(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(type(settings), 'renderer', lambda _: '/mock/musescore')
    commands = []
    def render(command, **kwargs):
        commands.append(command)
        assert kwargs['env']['QT_QPA_PLATFORM'] == 'offscreen'
        assert kwargs['timeout'] == settings.render_timeout_seconds
        destination = Path(command[2])
        if destination.suffix == '.pdf':
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=595, height=842)
            writer.write(destination)
        else:
            for page in [10, 1, 2]:
                destination.with_name(f'score-{page}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg"><text>{page}</text></svg>')
        return subprocess.CompletedProcess(command, 0, b'', b'')
    monkeypatch.setattr('app.services.score_render.subprocess.run', render)
    artifacts = render_score(tmp_path / 'score.musicxml', tmp_path, settings)
    assert len(commands) == 2
    assert [a['name'] for a in artifacts if a['media_type'] == 'image/svg+xml'] == ['page-1.svg', 'page-2.svg', 'page-3.svg']
    assert '>10<' in (tmp_path / 'page-3.svg').read_text()
    with zipfile.ZipFile(tmp_path / 'score-svgs.zip') as archive:
        assert archive.namelist() == ['page-1.svg', 'page-2.svg', 'page-3.svg']


@pytest.mark.parametrize('failure', ['exit', 'timeout', 'empty'])
def test_render_failure_is_reported(tmp_path, settings, monkeypatch, failure):
    monkeypatch.setattr(type(settings), 'renderer', lambda _: '/mock/musescore')
    def render(command, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 1 if failure == 'exit' else 0, b'', b'render failed')
    monkeypatch.setattr('app.services.score_render.subprocess.run', render)
    with pytest.raises(PipelineError):
        render_score(tmp_path / 'score.musicxml', tmp_path, settings)


def test_incomplete_svg_export_cannot_be_reported_as_completed(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(type(settings), 'renderer', lambda _: '/mock/musescore')
    def render(command, **kwargs):
        destination = Path(command[2])
        if destination.suffix == '.pdf':
            writer = PdfWriter()
            for _ in range(2):
                writer.add_blank_page(width=595, height=842)
            writer.write(destination)
        else:
            destination.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        return subprocess.CompletedProcess(command, 0, b'', b'')
    monkeypatch.setattr('app.services.score_render.subprocess.run', render)
    with pytest.raises(PipelineError, match='every SVG page'):
        render_score(tmp_path / 'score.musicxml', tmp_path, settings)
