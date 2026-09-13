import json
import subprocess

import pytest

from app.services.score_positions import attach_score_positions, read_positions


def fixture(tmp_path):
    pages = [tmp_path / f'page-{i}.svg' for i in (1, 2)]
    for page in pages:
        page.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1400"/>')
    source = tmp_path / 'score.spos'
    source.write_text('''<score><elements>
      <element id="0" x="1000" y="1400" sx="0" sy="2800" page="0"/>
      <element id="1" x="5000" y="1400" sx="0" sy="2800" page="0"/>
      <element id="2" x="2000" y="2800" sx="10" sy="1400" page="1"/>
    </elements><events><event elid="0" position="0"/>
      <event elid="1" position="1500"/><event elid="2" position="9000"/>
    </events></score>''')
    return source, pages


def test_segment_positions_use_real_timing_page_and_normalized_svg_coordinates(tmp_path):
    source, pages = fixture(tmp_path)
    assert read_positions(source, pages) == [
        {'time': 0, 'page': 0, 'x': .1, 'y': .1, 'height': .2},
        {'time': 1.5, 'page': 0, 'x': .5, 'y': .1, 'height': .2},
        {'time': 9, 'page': 1, 'x': .2, 'y': .2, 'height': .1},
    ]


@pytest.mark.parametrize('old,new', [('page="1"', 'page="8"'), ('x="1000"', 'x="nan"'),
                                    ('sy="2800"', 'sy="-1"'), ('position="9000"', 'position="-1"')])
def test_invalid_maps_are_rejected_instead_of_showing_a_misleading_cursor(tmp_path, old, new):
    source, pages = fixture(tmp_path)
    source.write_text(source.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_positions(source, pages)


def test_attachment_preserves_the_audio_and_adds_positions_atomically(tmp_path, settings, monkeypatch):
    source, _ = fixture(tmp_path)
    playback = {'duration': 10, 'tempo_bpm': 120, 'notes': [{'pitch': 60, 'start': 0, 'end': 2, 'velocity': 90}]}
    path = tmp_path / 'playback.json'
    path.write_text(json.dumps(playback))
    monkeypatch.setattr(type(settings), 'renderer', lambda _: '/mock/musescore')
    commands = []
    monkeypatch.setattr('app.services.score_positions.subprocess.run', lambda args, **kwargs: commands.append(args))
    assert attach_score_positions(tmp_path / 'score.musicxml', tmp_path, settings)
    result = json.loads(path.read_text())
    assert result.pop('positions')[-1]['time'] == 9
    assert result == playback
    assert commands[0][1:3] == ['-r', '300']
    assert not source.exists()
    assert not (tmp_path / 'playback-positions.tmp').exists()


def test_position_export_failure_keeps_working_audio(tmp_path, settings, monkeypatch):
    fixture(tmp_path)
    path = tmp_path / 'playback.json'
    original = '{"duration":10,"notes":[]}'
    path.write_text(original)
    monkeypatch.setattr(type(settings), 'renderer', lambda _: '/mock/musescore')
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('musescore', 1)
    monkeypatch.setattr('app.services.score_positions.subprocess.run', fail)
    assert not attach_score_positions(tmp_path / 'score.musicxml', tmp_path, settings)
    assert path.read_text() == original
