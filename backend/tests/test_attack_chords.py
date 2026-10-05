"""Chord engraving must not invent harmony, flatten arpeggios or shorten holds."""

import xml.etree.ElementTree as ET

import pretty_midi
import pytest
from app.models import ScoreOptions
from app.services.midi_to_score import _voices, midi_to_musicxml
from app.services.playback import _score_midi
from music21 import chord, converter


def export(tmp_path, notes, *, signature='4/4'):
    midi,xml=tmp_path/'chords.mid',tmp_path/'chords.musicxml'
    source=pretty_midi.PrettyMIDI(initial_tempo=120)
    part=pretty_midi.Instrument(0)
    part.notes=[pretty_midi.Note(v,p,s/2,e/2) for p,s,e,v in notes]
    source.instruments.append(part)
    source.write(str(midi))
    midi_to_musicxml(midi,xml,ScoreOptions(time_signature=signature), 'Supported chords')
    return converter.parse(str(xml)),ET.parse(xml)


def playback_events(score):
    played=_score_midi(score,120)
    return sorted((n.pitch,round(n.start*2,6),round(n.end*2,6),n.velocity)
                  for part in played.instruments for n in part.notes)


@pytest.mark.parametrize('signature',['4/4','3/4','6/8'])
def test_simultaneous_unequal_holds_are_chords_with_exact_playback(tmp_path,signature):
    notes=[(48,0.,6.,68),(52,0.,2.,55),(55,0.,3.,82),(72,1.,2.,96),(74,2.,3.,91)]
    score,tree=export(tmp_path,notes,signature=signature)
    first=score.parts[1].flatten().notes.first()
    assert isinstance(first,chord.Chord)
    assert [p.midi for p in first.pitches]==[48,52,55]
    assert playback_events(score)==sorted(notes)
    assert tree.findtext('.//scaling/millimeters')=='7'
    assert tree.find('.//defaults/system-layout') is None


def test_arpeggio_attacks_are_not_forced_into_a_chord(tmp_path):
    notes=[(48,0.,2.,70),(52,.5,2.5,60),(55,1.,3.,80)]
    score,_=export(tmp_path,notes)
    assert not any(isinstance(n,chord.Chord) for n in score.parts[1].flatten().notes)
    assert playback_events(score)==sorted(notes)


def test_later_repeated_chord_attacks_remain_distinct(tmp_path):
    notes=[(p,start,end,v) for start,end in ((0.,1.),(1.,2.),(2.,4.))
           for p,v in ((48,60),(52,76),(55,90))]
    score,_=export(tmp_path,notes)
    assert playback_events(score)==sorted(notes)
    assert len([n for n in score.parts[1].flatten().notes if isinstance(n,chord.Chord)])==3


def test_duplicate_same_key_attacks_with_different_holds_are_retained():
    voices=_voices({(0,12):{48:70},(0,24):{48:80,52:90}})
    attacks=[(span.start,p) for voice in voices for span in voice for p in span.pitches
             if p not in span.tied_from]
    assert sorted(attacks)==[(0,48),(0,48),(0,52)]


def test_held_chord_pitch_keeps_its_voice_when_melody_repeats(tmp_path):
    notes=[(64,0.,2.,54),(67,0.,8.,54),(64,2.,3.,92),(64,4.,5.,92)]
    score,_=export(tmp_path,notes)
    assert playback_events(score)==sorted(notes)


def test_overlapping_chords_with_shared_pitch_do_not_swap_ties(tmp_path):
    notes=[(60,0.,2.,70),(67,0.,6.,65),(64,1.,3.,90),(67,1.,8.,95)]
    score,_=export(tmp_path,notes)
    assert playback_events(score)==sorted(notes)
