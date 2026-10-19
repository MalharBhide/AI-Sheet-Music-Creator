"""Offline stage audit: authored notes and archived validation only.

No audio files, model inference, fitting, downloads, user jobs, or release gates.
Run with backend dependencies and PYTHONPATH=backend. Symbolic fixtures diagnose
transformations, not real-world transcription accuracy or perceptual quality.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pretty_midi
from scipy.optimize import linear_sum_assignment

from app.models import ScoreOptions
from app.services.audio_analysis import clean_notes, estimate_grid_phase, reduce_accompaniment
from app.services.midi_to_score import midi_to_musicxml
from app.services.playback import export_score_playback


def make_note(pitch, start, end, velocity=80):
    return pretty_midi.Note(velocity, pitch, float(start), float(end))


def rows(notes):
    return [dict(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity)
            for n in sorted(notes, key=lambda n: (n.start, n.pitch, n.end))]


def compare(reference, actual):
    """Pitch-specific one-to-one onset matching within 250 ms; not an F1 score.

    Report missing/extra events alongside errors, so deletion never earns zero
    timing error. Unmatched cases have null timing statistics, never zero.
    """
    paired = []
    for pitch in sorted({n.pitch for n in reference} | {n.pitch for n in actual}):
        expected = sorted((n for n in reference if n.pitch == pitch), key=lambda n: n.start)
        observed = sorted((n for n in actual if n.pitch == pitch), key=lambda n: n.start)
        if not expected or not observed:
            continue
        distances = np.abs(np.array([n.start for n in expected])[:, None]
                           - np.array([n.start for n in observed])[None, :])
        # Invalid assignments cost more than any collection of valid pairs.
        cost = np.where(distances <= .25, distances, 1000 * (len(expected) + len(observed)))
        left, right = linear_sum_assignment(cost)
        paired.extend((expected[i], observed[j]) for i, j in zip(left, right, strict=True)
                      if distances[i, j] <= .25)
    onset = [abs(a.start-b.start)*1000 for a, b in paired]
    release = [abs(a.end-b.end)*1000 for a, b in paired]
    spacing = []
    by_pitch = defaultdict(list)
    for a, b in paired:
        by_pitch[a.pitch].append((a, b))
    for pairs in by_pitch.values():
        pairs.sort(key=lambda pair: pair[0].start)
        spacing.extend(abs((a2.start-a1.start)-(b2.start-b1.start))*1000
                       for (a1, b1), (a2, b2) in zip(pairs, pairs[1:]))
    return dict(reference_notes=len(reference), actual_notes=len(actual), matched=len(paired),
                missing=len(reference)-len(paired), extra=len(actual)-len(paired),
                onset_mae_ms=float(np.mean(onset)) if onset else None,
                onset_max_ms=max(onset) if onset else None,
                release_mae_ms=float(np.mean(release)) if release else None,
                release_max_ms=max(release) if release else None,
                repeated_spacing_mae_ms=float(np.mean(spacing)) if spacing else None)


def fixtures():
    result = []

    def add(name, role, notes, interpretation):
        result.append(dict(name=name, role=role, notes=notes, interpretation=interpretation))

    add('straight_sixteenths_control', 'vocals',
        [make_note(64, i*.125, (i+1)*.125) for i in range(16)], 'Supported on-grid control.')
    add('complete_triplets_control', 'vocals',
        [make_note(64, i/6, (i+1)/6) for i in range(12)], 'Supported complete triplet control.')
    add('swing_repeats', 'vocals',
        [make_note(64, i*.5+start, i*.5+end) for i in range(4)
         for start, end in ((0, 1/3), (1/3, .5))],
        'Correct 2:1 swing; fixed straight grid lacks complete triplet evidence.')
    add('incomplete_triplets', 'vocals',
        [make_note(64, i*.5+start, i*.5+start+1/6) for i in range(4) for start in (0, 1/6)],
        'Correct two-note triplet fragments; no third attack to activate triplet rule.')
    add('fast_bass_32nds', 'bass',
        [make_note(48, i*.0625, (i+1)*.0625) for i in range(16)],
        'Correct fast bass attacks shorter than the 90 ms balanced cutoff.')
    add('staccato_bass', 'bass',
        [make_note(48, i*.25, i*.25+.07) for i in range(8)],
        'Correct 70 ms bass notes, with distinct attacks 250 ms apart.')
    add('soft_bass', 'bass',
        [make_note(48, i*.5, i*.5+.25, 28) for i in range(8)],
        'Correct quiet bass notes below the velocity floor; velocity is not confidence.')
    add('short_bass_releases', 'bass',
        [make_note(48, i*.25, i*.25+.095) for i in range(8)],
        'Correct staccato releases above the cutoff, rounded to a minimum grid duration.')
    add('tempo_change', 'vocals',
        [make_note(64, i*.25, (i+1)*.25) for i in range(16)]
        + [make_note(64, 4+i*.2, 4+(i+1)*.2) for i in range(16)],
        'Known 120-to-150 BPM change represented at a fixed 120 BPM score tempo.')
    add('independent_chord_holds_control', 'piano',
        [make_note(48, 0, 4), make_note(55, 0, 2), make_note(60, .5, 3),
         make_note(64, 1, 1.5), make_note(64, 2, 2.5)],
        'Supported independent chord holds and repeated-key control.')
    return result


def round_trip(notes, directory, name):
    source, xml, playback = [directory/f'{name}.{suffix}' for suffix in ('mid', 'musicxml', 'json')]
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    part = pretty_midi.Instrument(0)
    part.notes = notes
    midi.instruments.append(part)
    midi.write(str(source))
    midi_to_musicxml(source, xml, ScoreOptions(tempo_bpm=120), name)
    payload = export_score_playback(xml, source, playback, 120)
    return [make_note(n['pitch'], n['start'], n['end'], n['velocity']) for n in payload['notes']]


def archived_summary(root):
    directory = root/'training/results/approved-finetune-v70-v1'
    summary = {}
    for profile in ('tangent-quarter-margin', 'tangent-three-quarter-margin'):
        path = directory/profile/'diagnostic-100.json.gz'
        diagnostic = json.loads(gzip.decompress(path.read_bytes()))
        policies = {}
        for policy in ('raw', 'strict'):
            records = diagnostic[policy]['per_recording']
            corpora = {}
            for corpus in sorted({r['corpus'] for r in records}):
                subset = [r for r in records if r['corpus'] == corpus]
                pairs = sum(r['timing']['pairs'] for r in subset)
                errors = [sum(r['timing']['before_onset_release_error_sum'][i] for r in subset)
                          for i in (0, 1)]
                corpora[corpus] = dict(recordings=len(subset), matched_baseline_pairs=pairs,
                    baseline_matched_onset_mae_ms=1000*errors[0]/pairs if pairs else None,
                    baseline_matched_release_mae_ms=1000*errors[1]/pairs if pairs else None,
                    false_notes_removed=sum(r['false_notes_removed'] for r in subset),
                    failed_recordings=[r['id'] for r in subset if not r['passes']],
                    lost_reference_seconds=sum(r['coverage']['lost_reference_seconds'] for r in subset))
            policies[policy] = dict(corpora=corpora, passes=diagnostic[policy]['passes'],
                all_retained_clocks_identical=all(r['retained_clocks_identical'] for r in records))
        summary[profile] = dict(diagnostic_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                policies=policies)
    summary['training_corpora'] = json.loads((directory/'supervision.json').read_text())['training_corpora']
    summary['isolated_validation_corpora'] = dict(Counter(
        r['corpus'] for r in diagnostic['raw_pitch']['per_recording']))
    summary['limitations'] = ('Cached bass-candidate evaluation, before global quantization, arrangement, '
        'notation and playback. Matched-pair errors exclude missed notes and are not overall accuracy. '
        'Authored key-command timing is not a human-performance or acoustic-release label. '
        'No raw-versus-separated stage traces or full-song piano-arrangement reference is scored here.')
    return summary


def run(root, output):
    result = dict(version='offline-pipeline-diagnostic-v1', no_audio_read=True,
                  no_training=True, no_model_inference=True, production_modified=False, fixtures=[])
    with TemporaryDirectory(prefix='symbolic-pipeline-audit-') as folder:
        directory = Path(folder)
        for item in fixtures():
            reference = item.pop('notes')
            phase = estimate_grid_phase(reference, tempo_bpm=120, grid='sixteenth')
            shifted = [make_note(n.pitch, max(0, n.start-phase), max(0, n.end-phase), n.velocity)
                       for n in reference]
            cleaned = clean_notes(shifted, role=item['role'], detail='balanced', tempo_bpm=120,
                                  grid='sixteenth')
            direct = round_trip(reference, directory, item['name']+'-direct')
            final = round_trip(cleaned, directory, item['name']+'-cleaned')
            result['fixtures'].append({**item, 'phase_seconds': phase, 'reference': rows(reference),
                'after_cleanup': rows(cleaned), 'after_playback': rows(final),
                'cleanup_vs_reference': compare(reference, cleaned),
                'notation_only_vs_reference': compare(reference, direct),
                'playback_vs_cleanup': compare(cleaned, final),
                'playback_vs_reference': compare(reference, final)})
    lead = [make_note(72, 0, 2)]
    reference = [make_note(48, 0, 2, 80), make_note(55, 0, 2, 70)]
    candidates = reference + [make_note(52, 0, 2, 110)]
    selected = reduce_accompaniment(candidates, detail='balanced', melody=lead)
    result['arrangement_probe'] = dict(
        interpretation='Authored adversarial candidates: long loud wrong tone competes with two correct voices. '
        'Shows heuristic behavior, not frequency of this error in recordings.',
        reference=rows(reference), candidates=rows(candidates), selected=rows(selected),
        candidate_vs_reference=compare(reference, candidates), selected_vs_reference=compare(reference, selected),
        clean_control=compare(reference, reduce_accompaniment(reference, detail='balanced', melody=lead)))
    result['archived_validation'] = archived_summary(root)
    files = ['backend/app/services/'+name+'.py' for name in
             ('audio_analysis', 'piano_transcription', 'midi_to_score', 'playback',
              'source_separation', 'left_presence_network')]
    files.append('scripts/diagnose_transcription_pipeline.py')
    result['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), fixtures=len(result['fixtures']), no_audio_read=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
