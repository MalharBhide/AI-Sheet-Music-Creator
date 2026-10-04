"""Compare rhythm/arrangement decoding on original symbolic note fixtures only."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from app.services import audio_analysis as current


def note(pitch, start, end, velocity=90):
    return SimpleNamespace(pitch=pitch, start=start, end=end, velocity=velocity)


def decode(module, notes, bpm, refine=False):
    used = module.refine_repeat_tempo(notes, bpm) if refine else bpm
    offset = module.estimate_grid_phase(notes, tempo_bpm=used, grid='sixteenth')
    shifted = [note(n.pitch, max(0., n.start - offset), max(0., n.end - offset), n.velocity) for n in notes]
    result = module.clean_notes(shifted, role='vocals', detail='balanced', tempo_bpm=used, grid='sixteenth')
    return used, result


def benchmark(output, baseline, baseline_source=None):
    if output.exists():
        raise ValueError('Preserve completed benchmark reports')
    source = baseline_source.read_text() if baseline_source else subprocess.run(
        ['git', 'show', f'{baseline}:backend/app/services/audio_analysis.py'],
        capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'baseline.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('fixture_baseline', path)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        rows = []
        for bpm in (80., 120., 160.):
            for rhythm, period, offset, estimated in (
                ('shifted repeated triplets', 60 / bpm / 3, .12 * 60 / bpm, bpm),
                ('small tempo error on repeated quarters', 60 / bpm, .06, bpm * 1.01),
            ):
                notes = [note(64, offset + i * period, offset + (i + .8) * period) for i in range(48)]
                row = {'fixture': rhythm, 'reference_tempo': bpm, 'reference_interval': period}
                for name, module, refine in [('baseline', previous, False), ('updated', current, True)]:
                    used, result = decode(module, notes, estimated, refine)
                    spacing = np.diff([n.start for n in result])
                    row[name] = {'notes': len(result), 'tempo': used,
                                 'mean_spacing_error_ms': float(np.mean(np.abs(spacing - period)) * 1000),
                                 'maximum_spacing_error_ms': float(np.max(np.abs(spacing - period)) * 1000)}
                assert row['updated']['notes'] == len(notes)
                assert row['updated']['maximum_spacing_error_ms'] < 1e-6
                rows.append(row)
        lead = [note(p, .5 * i, .5 * (i + 1)) for i, p in enumerate([72, 74, 76, 74, 72, 79])]
        echo = [note(n.pitch - 12, n.start, n.end, 75) for n in lead]
        support = [note(48, 0, 3, 70)]
        arrangement = {name: len(module.reduce_accompaniment(echo + support, detail='balanced', melody=lead))
                       for name, module in [('baseline', previous), ('updated', current)]}
        assert arrangement == {'baseline': 7, 'updated': 1}
        result = {'baseline_revision': baseline, 'baseline_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'passes': True, 'timing': rows,
                  'backing_notes_in_octave_copy_fixture': arrangement, 'lead_notes': len(lead),
                  'scope': 'Original symbolic detections, not audio-model accuracy. No user audio or score generation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', default='b9ee29e')
    parser.add_argument('--baseline-source', type=Path, help='Exact source exported from the baseline revision when git is unavailable in the runtime')
    args = parser.parse_args()
    benchmark(args.output, args.baseline, args.baseline_source)
