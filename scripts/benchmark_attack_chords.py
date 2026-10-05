"""Native chord engraving check on original symbolic piano phrases only."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pretty_midi
from app.config import Settings
from app.models import ScoreOptions
from app.services import midi_to_score as current
from app.services.playback import _score_midi, export_score_playback
from app.services.score_render import render_score
from music21 import chord, converter, stream
from pypdf import PdfReader


def phrase():
    notes=[]
    for bar,root in enumerate((48,45,53,55)):
        start=bar*4.
        notes.extend([(root,start,start+4.,62),(root+4,start,start+2.,52),
                      (root+7,start,start+3.,58)])
        notes.extend((72+offset,start+beat,start+beat+.5,92)
                     for beat,offset in enumerate((0,2,4,2)))
    return notes


def verify(output,baseline_source):
    output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('before_chord_engraving',baseline_source)
    baseline=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=baseline
    spec.loader.exec_module(baseline)
    notes=phrase()
    results={}
    for name,module in (('before',baseline),('after',current)):
        work=output/name
        work.mkdir()
        midi,xml=work/'transcription.mid',work/'score.musicxml'
        source=pretty_midi.PrettyMIDI(initial_tempo=120)
        piano=pretty_midi.Instrument(0)
        piano.notes=[pretty_midi.Note(v,p,s/2,e/2) for p,s,e,v in notes]
        source.instruments.append(piano)
        source.write(str(midi))
        module.midi_to_musicxml(midi,xml,ScoreOptions(), 'Chords with a clear melody')
        parsed=converter.parse(str(xml))
        played=_score_midi(parsed,120)
        actual=sorted((n.pitch,round(n.start*2,6),round(n.end*2,6),n.velocity)
                      for part in played.instruments for n in part.notes)
        assert actual==sorted(notes)
        payload=export_score_playback(xml,midi,work/'playback.json',120)
        artifacts=render_score(xml,work,Settings(storage_root=work,_env_file=None))
        playback=json.loads((work/'playback.json').read_text())
        assert playback['positions'] and {p['page'] for p in playback['positions']}=={0}
        for position in playback['positions']:
            # Engraved rests and held-note tie boundaries also have positions.
            assert any(abs(position['time']-n[boundary])<.002
                       for n in payload['notes'] for boundary in ('start','end'))
        left=parsed.parts[1]
        results[name]={'notes':len(actual),'exact_pitches_attacks_releases_velocities':True,
                       'left_hand_chords':sum(isinstance(n,chord.Chord) for n in left.flatten().notes),
                       'maximum_left_hand_voices':max(max(1,len(m.voices)) for m in left.getElementsByClass(stream.Measure)),
                       'pdf_pages':len(PdfReader(work/'score.pdf').pages),'artifacts':len(artifacts),
                       'sheet_audio_position_clock_verified':True}
    assert results['after']['left_hand_chords']>results['before']['left_hand_chords']
    assert results['after']['maximum_left_hand_voices']<results['before']['maximum_left_hand_voices']
    assert results['after']['pdf_pages']<=results['before']['pdf_pages']
    report={'passes':True,'before_source_sha256':hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
            'after_source_sha256':hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
            'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results,
            'original_note_size_and_layout_preserved':True,'no_user_audio_or_scores':True,
            'scope':'Original four-bar symbolic piano phrase, native XML/PDF/SVG/playback. No model accuracy claim.'}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('baseline_source',type=Path)
    args=parser.parse_args()
    verify(args.output,args.baseline_source)
