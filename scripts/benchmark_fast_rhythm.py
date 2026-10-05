"""Compare V16/new score timing on original, labeled symbolic fixtures only."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from app.services import audio_analysis as current


def events(bpm, division, phase=0., count=48):
    period = 60/bpm/division
    return [SimpleNamespace(pitch=64, start=phase+i*period,end=phase+(i+.98)*period,velocity=90)
            for i in range(count)]


def decoded(module, items, bpm):
    phase = module.estimate_grid_phase(items, tempo_bpm=bpm, grid='sixteenth')
    shifted = [SimpleNamespace(**{**vars(n),'start':max(0,n.start-phase),'end':n.end-phase}) for n in items]
    return module.clean_notes(shifted, role='vocals', detail='balanced', tempo_bpm=bpm, grid='sixteenth')


def benchmark(output, source):
    if output.exists():
        raise ValueError('Preserve completed repeat timing benchmarks')
    spec = importlib.util.spec_from_file_location('v16_rhythm_baseline', source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rows = []
    for bpm in (80.,120.,160.):
        for phase in (0.,.017):
            items = events(bpm,8,phase)
            old, new = [decoded(module,items,bpm) for module in (baseline,current)]
            errors = np.abs(np.diff([n.start for n in new])-60/bpm/8)
            assert len(new)==48 and float(errors.max())<1e-9
            rows.append({'bpm':bpm,'phase_seconds':phase,'reference_attacks':48,
                         'baseline_attacks':len(old),'updated_attacks':len(new),
                         'maximum_updated_spacing_error_ms':float(errors.max())*1000})
    controls = []
    for bpm in (80.,120.,160.):
        for division in (1,2,3,4):
            items = events(bpm,division)
            old,new = [decoded(module,items,bpm) for module in (baseline,current)]
            before,after = [np.asarray([[n.start,n.end,n.pitch,n.velocity] for n in notes]) for notes in (old,new)]
            np.testing.assert_array_equal(before,after)
            controls.append({'bpm':bpm,'division':division,'notes':len(new),'unchanged':True})
    result = {'passes':True,'baseline_revision':'9eac730','baseline_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'updated_source_sha256':hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
              'repeated_notes':rows,'straight_triplet_controls':controls,
              'scope':'Original symbolic detections with known attacks, not audio-model accuracy or user songs. Raw pitch/count unchanged; compare score decoding. No uploads or saved scores processed.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passes':True,'repeat_cases':len(rows),'unchanged_controls':len(controls)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    benchmark(args.output,args.source)
