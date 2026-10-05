"""Cache actual V11 intervals; re-infer datasets only when a duration changed."""

import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import soundfile as sf
from app.services.bass_articulation import BassArticulation
from app.services.bass_verifier import BassVerifier
from app.services.note_evidence import CONTEXT_VERSION, note_features
from bass_training_data import NAMES, eligible, identity, load
from current_bass_v11_baseline import VERSION, decisions, hashes
from prepare_robust_training_stems import digest, preserve

_MODEL = None


def native_evidence(item, decision):
    global _MODEL
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    if _MODEL is None:
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    samples, rate = sf.read(item['audio'], dtype='float32')
    if rate != 22050 or samples.ndim != 1 or len(samples) > rate * 30 or not np.isfinite(samples).all():
        raise ValueError('Changed bounded dataset waveform clock')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        acoustic, midi, _ = predict(item['audio'], model_or_model_path=_MODEL,
            minimum_frequency=float(pretty_midi.note_number_to_hz(21)),
            maximum_frequency=float(pretty_midi.note_number_to_hz(60)),
            onset_threshold=.5, frame_threshold=.3, minimum_note_length=90.,
            multiple_pitch_bends=False, melodia_trick=False)
        notes = [note for part in midi.instruments for note in part.notes]
        first, stop = item['evaluation_window']
        notes = [note for note in notes if first <= note.start < stop]
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes]).reshape(-1, 4)
        np.testing.assert_array_equal(events, item['events'], err_msg=item['id'])
        raw_x = note_features(samples, rate, acoustic, notes, include_context=True)
        np.testing.assert_array_equal(raw_x, item['x'], err_msg=item['id'])
        # Exercise both actual runtime filters on independent native note objects.
        native = SimpleNamespace(instruments=[SimpleNamespace(notes=notes)])
        bass = BassVerifier()
        bass.filter(item['audio'], acoustic, native)
        merges = BassArticulation().filter(item['audio'], acoustic, native, bass)
        output_notes = native.instruments[0].notes
        actual = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in output_notes]).reshape(-1, 4)
        np.testing.assert_array_equal(actual, decision['events'], err_msg=item['id'])
        if merges != decision['merged_boundaries']:
            raise ValueError('Changed native V11 merges')
        x = note_features(samples, rate, acoustic, output_notes, include_context=True)
    return x


def prepare(roots, regressions, output):
    output.mkdir(exist_ok=True)
    all_roots = roots + regressions
    if len(set(all_roots)) != len(all_roots):
        raise ValueError('Duplicate V11 cache root')
    plan = {'version': VERSION, 'baseline_hashes': hashes(),
            'fitting_manifests': {str(root): digest(root / 'manifest.json') for root in roots},
            'consumed_regression_manifests': {str(root): digest(root / 'manifest.json') for root in regressions},
            'code_sha256': digest(Path(__file__)), 'feature_version': CONTEXT_VERSION,
            'features': list(NAMES), 'no_user_audio_or_scores': True,
            'scope': 'Reuse per-note independent evidence only for unchanged intervals; recompute actual native BP/CQT evidence for every merged interval. Test caches remain consumed regression only.'}
    preserve(output / 'plan.json', plan)
    work = output / 'features'
    work.mkdir(exist_ok=True)
    verifier, records, ids, source_groups = BassArticulation(), [], set(), {'train': set(), 'validation': set(), 'test': set()}
    for root in all_roots:
        groups = ('train', 'validation') if root in roots else ('test',)
        manifest = json.loads((root / 'manifest.json').read_text())
        original_items = {item['id']: item for item in manifest['items']}
        if manifest['plan_sha256'] != digest(root / 'plan.json') or any(i['group'] not in groups for i in manifest['items']):
            raise ValueError('Stale or mixed V11 cache partitions')
        for group in groups:
            for item in load(root, group):
                if item['id'] in ids:
                    raise ValueError('Duplicate V11 cache identity')
                ids.add(item['id'])
                source_groups[group].add(item['source_group'])
                with np.load(root / 'features' / (item['id'] + '.npz'), allow_pickle=False) as saved:
                    duration = float(saved['duration'])
                if sf.info(item['audio']).duration != duration:
                    raise ValueError('Changed V11 cache audio duration')
                decision = decisions(item, duration, verifier)
                record = {'id': item['id'], 'group': group, 'source_group': item['source_group'],
                          'corpus': item['corpus'], 'root': str(root), 'identity': identity(original_items[item['id']]),
                          'source_cache_sha256': manifest['cache_sha256'][item['id']],
                          'audio_sha256': digest(Path(item['audio'])),
                          'raw_candidates': decision['raw_count'], 'retained_v11': len(decision['events']),
                          'merged_boundaries': decision['merged_boundaries'],
                          'recomputed_intervals': int(decision['changed_intervals'].sum())}
                path = work / (item['id'] + '.npz')
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if str(saved['identity']) != json.dumps(record, sort_keys=True):
                            raise ValueError('Changed completed V11 cache identity')
                        np.testing.assert_array_equal(saved['events'], decision['events'])
                        np.testing.assert_array_equal(saved['raw_indices'], decision['raw_indices'])
                else:
                    x = native_evidence(item, decision) if decision['merged_boundaries'] else item['x'][decision['raw_indices']]
                    bass = BassVerifier()
                    p, g = [model.probability(x[:, :model.feature_count]) for model in bass.models]
                    np.savez_compressed(path, x=x, events=decision['events'], raw_indices=decision['raw_indices'],
                                        eligible=eligible(decision['events'], duration), context=p, guardian=g,
                                        duration=duration, identity=json.dumps(record, sort_keys=True),
                                        baseline_version=VERSION, feature_version=CONTEXT_VERSION,
                                        plan_sha256=digest(output / 'plan.json'))
                record['cache_sha256'] = digest(path)
                records.append(record)
                print(json.dumps({'cached': len(records), 'id': item['id'], 'merges': decision['merged_boundaries']}), flush=True)
    if any(source_groups[a] & source_groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('V11 source groups leak between partitions')
    preserve(output / 'manifest.json', {'version': VERSION, 'plan_sha256': digest(output / 'plan.json'),
             'items': records, 'no_user_audio_or_scores': True, 'all_tests_consumed_regression': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--regression', type=Path, action='append', default=[])
    args = parser.parse_args()
    prepare(args.roots, args.regression, args.output)
