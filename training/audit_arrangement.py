"""Inspect lead retention and masking on a bounded normalized audio excerpt."""

import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi

from app.services.audio_analysis import (
    arrange_melody_register,
    balance_piano_arrangement,
    clean_notes,
    preserve_melody_releases,
    reduce_accompaniment,
)
from app.services.piano_transcription import _GeneralEngine
from app.services.source_separation import StemSeparator


def summarize(notes):
    if not notes:
        return {'notes': 0}
    return {'notes': len(notes), 'median_pitch': float(np.median([n.pitch for n in notes])),
            'below_middle_c': float(np.mean([n.pitch < 60 for n in notes])),
            'sounding_seconds': float(sum(n.end - n.start for n in notes)),
            'median_duration': float(np.median([n.end - n.start for n in notes]))}


def run(path, output, tempo, grid='sixteenth', reuse=False):
    output.mkdir(parents=True, exist_ok=True)
    if reuse:
        sources = {role: output / f'{role}-raw.mid' for role in ('vocals', 'bass', 'other')
                   if (output / f'{role}-raw.mid').exists()}
    else:
        separator, engine = StemSeparator(), _GeneralEngine('balanced')
        sources = separator.separate(path, output)
    report, parts = {}, {}
    for role, source in sources.items():
        midi = pretty_midi.PrettyMIDI(str(source)) if reuse else engine.predict(source, role, tempo)
        notes = [n for part in midi.instruments for n in part.notes]
        if not reuse:
            midi.write(str(output / f'{role}-raw.mid'))
        part = pretty_midi.Instrument(0, name=role)
        part.notes = clean_notes(notes, role=role, detail='balanced', tempo_bpm=tempo, grid=grid)
        if role == 'other':
            part.notes = reduce_accompaniment(part.notes, detail='balanced')
        parts[role] = part
        report[role] = {'raw': summarize(notes), 'score': summarize(part.notes)}
    shift = arrange_melody_register(parts)
    support_changes = preserve_melody_releases(parts)
    balance_piano_arrangement(parts)
    for role, part in parts.items():
        report[role]['arranged'] = summarize(part.notes)
    report['arrangement'] = {'grid': grid, 'melody_semitone_shift': shift,
                              'support_notes_changed_for_melody': support_changes,
                              'lead_after_register': summarize(parts['vocals'].notes) if 'vocals' in parts else {}}
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi.instruments.extend(parts.values())
    midi.write(str(output / f'arrangement-{grid}.mid'))
    (output / f'audit-{grid}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tempo', type=float, default=120)
    parser.add_argument('--grid', choices=['eighth', 'sixteenth'], default='sixteenth')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    run(args.path, args.output, args.tempo, args.grid, args.reuse)
