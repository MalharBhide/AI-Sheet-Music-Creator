"""Fast repeated attacks must survive cleanup, notation and playback timing."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
from app.models import ScoreOptions
from app.services.audio_analysis import clean_notes, estimate_grid_phase, fast_beats
from app.services.midi_to_score import midi_to_musicxml
from music21 import converter


def notes(bpm, count=48, phase=0., division=8, jitter=0.):
    period = 60/bpm/division
    return [SimpleNamespace(pitch=64, start=phase+i*period+jitter*(-1)**i,
                            end=phase+(i+.98)*period, velocity=90) for i in range(count)]


@pytest.mark.parametrize('bpm', [80., 120., 160.])
@pytest.mark.parametrize('phase', [0., .017])
def test_fast_repeated_keys_keep_all_attacks_and_equal_intervals(bpm, phase):
    original = notes(bpm, phase=phase)
    offset = estimate_grid_phase(original, tempo_bpm=bpm, grid='sixteenth')
    shifted = [SimpleNamespace(**{**vars(n), 'start': max(0, n.start-offset), 'end': n.end-offset}) for n in original]
    result = clean_notes(shifted, role='vocals', detail='balanced', tempo_bpm=bpm, grid='sixteenth')
    assert len(result) == len(original)
    np.testing.assert_allclose(np.diff([n.start for n in result]), 60/bpm/8, atol=1e-10)
    assert all(n.pitch == 64 for n in result)
    assert all(n.end <= next_n.start + 1e-10 for n, next_n in zip(result[:-1], result[1:], strict=True))


def test_fast_grid_requires_real_distinct_adjacent_subdivisions():
    chord = [SimpleNamespace(pitch=p, start=.0625, end=.5, velocity=90) for p in range(48, 85)]
    assert fast_beats(chord, 120) == set()
    assert fast_beats(notes(120, count=3), 120) == set()
    assert fast_beats(notes(120, division=4), 120) == set()
    # Three spaced off-grid attacks do not support four consecutive slots.
    sparse = [SimpleNamespace(pitch=64, start=t, end=t+.1, velocity=90) for t in (.0625,.1875,.3125,.4375)]
    assert fast_beats(sparse, 120) == set()
    jittered = notes(120, count=8, jitter=.025)
    assert fast_beats(jittered, 120) == set()


def test_simple_grid_stays_coarse_and_long_holds_remain_intact():
    fast = notes(120)
    simple = clean_notes(fast, role='vocals', detail='balanced', tempo_bpm=120, grid='eighth')
    assert len(simple) < len(fast)
    held = SimpleNamespace(pitch=48, start=0., end=4., velocity=90)
    result = clean_notes(fast+[held], role='piano', detail='balanced', tempo_bpm=120, grid='sixteenth')
    assert [(n.start,n.end) for n in result if n.pitch==48] == [(0.,4.)]


@pytest.mark.parametrize('bpm', [80., 120., 160.])
def test_fast_repeated_keys_survive_musicxml_and_midi_round_trip(tmp_path, bpm):
    raw = notes(bpm)
    cleaned = clean_notes(raw, role='vocals', detail='balanced', tempo_bpm=bpm, grid='sixteenth')
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    part = pretty_midi.Instrument(0)
    part.notes = [pretty_midi.Note(n.velocity,n.pitch,n.start,n.end) for n in cleaned]
    midi.instruments.append(part)
    source, xml, playback = tmp_path/'source.mid', tmp_path/'score.musicxml', tmp_path/'playback.mid'
    midi.write(str(source))
    midi_to_musicxml(source, xml, ScoreOptions(tempo_bpm=bpm), 'Original repeat fixture')
    tree = ET.parse(xml)
    assert len(tree.findall('.//note/pitch')) == 48
    assert set(n.text for n in tree.findall('.//note/type')) >= {'32nd'}
    # Engraving uses the original size, without injected staff/page scaling.
    assert not tree.findall('.//staff-layout') and not tree.findall('.//system-layout')
    score = converter.parse(str(xml))
    score.write('midi', fp=str(playback))
    result = pretty_midi.PrettyMIDI(str(playback)).instruments[0].notes
    assert len(result) == 48
    np.testing.assert_allclose(np.diff([n.start for n in result]), 60/bpm/8, atol=5e-6)
