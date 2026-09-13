import numpy as np
import pretty_midi
import pytest

from app.services.audio_analysis import clean_notes
from app.services.vocal_refinement import refine_notes


def note(pitch, start=0, end=1, velocity=80):
    return pretty_midi.Note(velocity, pitch, start, end)


def test_late_harmonic_does_not_interrupt_a_supported_vocal_note():
    detected = [note(57), note(69, .25, .75, 60)]
    refined = refine_notes(detected, np.full(100, 57.), np.ones(100), .01)
    score = clean_notes(refined, role='vocals', detail='balanced', tempo_bpm=120,
                        grid='sixteenth')
    assert [(n.pitch, n.start, n.end) for n in score] == [(57, 0, 1)]


def test_real_octave_jump_and_repeated_attacks_are_preserved():
    pitches = np.repeat([57., 69., 69.], 50)
    detected = [note(57, 0, .5), note(69, .5, 1), note(69, 1, 1.5)]
    refined = refine_notes(detected, pitches, np.ones(150), .01)
    assert [(n.pitch, n.start, n.end) for n in refined] == [
        (57, 0, .5), (69, .5, 1), (69, 1, 1.5)]


def test_unsupported_outer_tail_is_trimmed_to_voiced_audio():
    pitches = np.r_[np.full(60, 60.), np.full(140, np.nan)]
    refined = refine_notes([note(60, 0, 2)], pitches, np.ones(200), .01)
    assert refined[0].start == 0
    assert refined[0].end == pytest.approx(.61)


def test_uncertain_pitch_tracking_keeps_neural_notes():
    detected = [note(60)]
    for pitches, confidence in [(np.full(100, np.nan), np.zeros(100)),
                                (np.full(100, 72.), np.full(100, .1))]:
        assert refine_notes(detected, pitches, confidence, .01) == detected


def test_vibrato_does_not_change_pitch_or_split_the_note():
    pitches = 60 + .6 * np.sin(np.arange(100) / 3)
    refined = refine_notes([note(60)], pitches, np.ones(100), .01)
    assert [(n.pitch, n.start, n.end) for n in refined] == [(60, 0, 1)]


def test_empty_short_and_outside_track_events_do_not_crash():
    assert refine_notes([], np.array([]), np.array([]), .01) == []
    detected = [note(60, 0, .02), note(60, 2, 3)]
    assert refine_notes(detected, np.full(4, 60.), np.ones(4), .01) == detected


def test_continuous_voiced_pitch_merges_false_attacks_without_extending_the_end():
    refined = refine_notes([note(60, 0, .5), note(60, .5, 1)],
                           np.full(100, 60.), np.ones(100), .01, np.ones(100))
    assert [(n.start, n.end) for n in refined] == [(0, 1)]


def test_new_amplitude_attack_preserves_rearticulation_of_the_same_pitch():
    rms = np.ones(100)
    rms[46:50] = .3
    refined = refine_notes([note(60, 0, .5), note(60, .5, 1)],
                           np.full(100, 60.), np.ones(100), .01, rms)
    assert [(n.start, n.end) for n in refined] == [(0, .5), (.5, 1)]


def test_breath_between_same_pitch_notes_preserves_both_attacks():
    confidence = np.ones(100)
    confidence[47:52] = 0
    refined = refine_notes([note(60, 0, .5), note(60, .5, 1)],
                           np.full(100, 60.), confidence, .01, np.ones(100))
    assert len(refined) == 2
