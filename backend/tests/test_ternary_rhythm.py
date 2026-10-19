"""Repeated swing/partial-triplet evidence must survive cleanup and playback."""

from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest

from app.models import ScoreOptions
from app.services.audio_analysis import clean_notes, estimate_grid_phase, triplet_beats
from app.services.midi_to_score import midi_to_musicxml
from app.services.playback import export_score_playback


def pair_notes(bpm=120, slot=2, beats=6, phase=0., jitter=0.):
    beat = 60/bpm
    result = []
    for index in range(beats):
        for offset, end in ((0., slot/3), (slot/3, 1. if slot == 2 else 2/3)):
            start = max(0., phase+(index+offset)*beat+jitter*(1 if offset else -1))
            result.append(SimpleNamespace(pitch=64, start=start,
                                          end=phase+(index+end)*beat, velocity=80))
    return result


def clean(notes, bpm=120, role='vocals', grid='sixteenth'):
    phase = estimate_grid_phase(notes, tempo_bpm=bpm, grid=grid)
    shifted = [SimpleNamespace(**{**vars(n), 'start': max(0., n.start-phase),
                                  'end': max(n.start-phase, n.end-phase)}) for n in notes]
    return clean_notes(shifted, role=role, detail='balanced', tempo_bpm=bpm, grid=grid)


@pytest.mark.parametrize('bpm', [80., 120., 160.])
@pytest.mark.parametrize('slot', [1, 2])
@pytest.mark.parametrize('phase', [0., .017])
@pytest.mark.parametrize('role', ['vocals', 'bass', 'piano'])
def test_repeated_pairs_preserve_pitch_attack_spacing_and_releases(bpm, slot, phase, role):
    reference = pair_notes(bpm, slot)
    supplied = pair_notes(bpm, slot, phase=phase)
    if role == 'bass':
        for n in reference + supplied:
            n.pitch = 48
    original = [vars(n).copy() for n in supplied]
    result = clean(supplied, bpm, role)
    assert len(result) == len(reference)
    assert [n.pitch for n in result] == [n.pitch for n in reference]
    np.testing.assert_allclose([n.start for n in result], [n.start for n in reference], atol=1e-9)
    np.testing.assert_allclose([n.end for n in result], [n.end for n in reference], atol=1e-9)
    assert [vars(n) for n in supplied] == original


@pytest.mark.parametrize('slot', [1, 2])
def test_small_onset_jitter_does_not_turn_repeated_pairs_into_straight_rhythm(slot):
    notes = pair_notes(slot=slot, phase=.017, jitter=.004)
    result = clean(notes)
    np.testing.assert_allclose(np.diff([n.start for n in result]),
                               np.diff([n.start for n in pair_notes(slot=slot)]), atol=1e-9)


def test_one_or_two_pairs_do_not_establish_ternary_rhythm():
    for slot in (1, 2):
        for beats in (1, 2):
            assert triplet_beats(pair_notes(slot=slot, beats=beats), 120) == set()


def test_chord_copies_and_repeated_pairs_across_rests_are_insufficient():
    copies = [SimpleNamespace(**{**vars(n), 'pitch': pitch})
              for n in pair_notes(beats=1) for pitch in range(40, 80)]
    assert triplet_beats(copies, 120) == set()
    notes = pair_notes(beats=2) + pair_notes(beats=2, phase=2.)
    assert triplet_beats(notes, 120) == set()


def test_straight_subdivisions_and_unstable_pairs_do_not_become_triplets():
    for division in (2, 4, 8):
        notes = [SimpleNamespace(pitch=64, start=i*.5/division,
                                 end=(i+1)*.5/division, velocity=80) for i in range(division*6)]
        assert triplet_beats(notes, 120) == set()
    alternating = pair_notes(slot=1, beats=1) + pair_notes(slot=2, beats=1, phase=.5)
    alternating += pair_notes(slot=1, beats=1, phase=1.)
    assert triplet_beats(alternating, 120) == set()
    assert triplet_beats(pair_notes(jitter=.025), 120) == set()


def test_sparse_32nd_pairs_are_not_phase_shifted_to_look_like_triplets():
    notes = [SimpleNamespace(pitch=64, start=.017+(i+slot)*.5,
                             end=.017+(i+slot)*.5+.1, velocity=80)
             for i in range(6) for slot in (0., .375)]
    # A ternary phase could make these pairs look approximately ternary. The
    # exact 32nd-grid fit wins, even without enough attacks to enable fast_beats.
    assert estimate_grid_phase(notes, tempo_bpm=120, grid='sixteenth') == 0.


def test_supported_run_does_not_retime_neighboring_straight_passages_or_long_holds():
    notes = pair_notes(beats=3, phase=1.)
    notes += [SimpleNamespace(pitch=72, start=t, end=t+.125, velocity=90)
              for t in (0., .125, .25, .375, 2.5, 2.625, 2.75, 2.875)]
    notes += [SimpleNamespace(pitch=48, start=0., end=4., velocity=80)]
    assert triplet_beats(notes, 120) == {2, 3, 4}
    result = clean_notes(notes, role='piano', detail='balanced', tempo_bpm=120, grid='sixteenth')
    assert len(result) == len(notes)
    np.testing.assert_allclose(sorted((n.pitch, n.start, n.end) for n in result),
                               sorted((n.pitch, n.start, n.end) for n in notes), atol=1e-9)


def test_simple_grid_and_detection_thresholds_are_unchanged():
    result = clean(pair_notes(), grid='eighth')
    assert all(n.start % .25 == 0 for n in result)
    # This patch does not weaken acoustic/cleanup eligibility to add notes.
    quiet = [SimpleNamespace(**{**vars(n), 'velocity': 28}) for n in pair_notes()]
    assert clean(quiet, role='bass') == []
    short = [SimpleNamespace(**{**vars(n), 'end': n.start+.07}) for n in pair_notes()]
    assert clean(short, role='bass') == []


@pytest.mark.parametrize('slot', [1, 2])
@pytest.mark.parametrize('length', [.095, .125])
def test_partial_triplet_attack_pattern_does_not_require_legato_holds(slot, length):
    notes = pair_notes(slot=slot)
    for n in notes:
        n.end = n.start + length
    result = clean(notes)
    assert len(result) == len(notes)
    # Eighth-triplet attack spacing must not impose a 166.7 ms minimum hold.
    assert all(n.end-n.start <= .125 + 1e-9 for n in result)
    assert max(abs(a.end-b.end) for a, b in zip(notes, result, strict=True)) <= 1/48 + 1e-9


@pytest.mark.parametrize('bpm', [80., 120., 160.])
@pytest.mark.parametrize('slot', [1, 2])
@pytest.mark.parametrize('short', [False, True])
def test_correct_spacing_survives_actual_score_playback(tmp_path, bpm, slot, short):
    reference = pair_notes(bpm, slot)
    supplied = pair_notes(bpm, slot, phase=.017)
    if short:
        for n in reference + supplied:
            n.end = n.start + 60/bpm/4
    result = clean(supplied, bpm)
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    part = pretty_midi.Instrument(0)
    part.notes = result
    midi.instruments.append(part)
    source, xml, playback = tmp_path/'notes.mid', tmp_path/'notes.musicxml', tmp_path/'playback.json'
    midi.write(str(source))
    midi_to_musicxml(source, xml, ScoreOptions(tempo_bpm=bpm), 'Ternary rhythm fixture')
    payload = export_score_playback(xml, source, playback, bpm)
    assert len(payload['notes']) == len(reference)
    np.testing.assert_allclose([n['start'] for n in payload['notes']],
                               [n.start for n in reference], atol=2e-6)
    np.testing.assert_allclose([n['end'] for n in payload['notes']],
                               [n.end for n in reference], atol=2e-6)
