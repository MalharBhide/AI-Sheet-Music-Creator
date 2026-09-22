"""Check deployed filter parity on cached features without audio transcription."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from evaluate_context_correction import previous_events
from train_note_verifier import load_data, write
from train_residual_verifier import cached_external, prepare_items, residual_keep

from app.services import accompaniment_verifier as service


def verify(directory, output):
    if output.exists():
        raise ValueError('Preserve the runtime parity report')
    items = load_data(directory, 'validation') + load_data(directory, 'test')
    for name in ('note-verifier-oxford', 'note-verifier-oxford-extended', 'note-verifier-stems-test',
                 'later-vienna-context', 'oxford-later-context', 'residual-guitar-tails'):
        base = directory.parent if name.startswith('note-verifier-') else directory
        items.extend(cached_external(directory, directory.parent / name / 'manifest.json', base))
    torch.set_num_threads(2)
    verifier = service.AccompanimentVerifier()
    prepared = prepare_items(directory, items, verifier)
    total, rejected = 0, 0
    for item in prepared:
        features = item['x'].copy()
        # Unshared context is unused, but original V2 features still matter.
        features[:, :26] = previous_events(directory, item)[0]
        notes = [SimpleNamespace(start=float(s), end=float(e), pitch=int(p), velocity=int(v))
                 for s, e, p, v in item['events']]
        attributes = [vars(note).copy() for note in notes]
        split = len(notes) // 2
        parts = [SimpleNamespace(notes=notes[:split]), SimpleNamespace(notes=notes[split:])]
        midi = SimpleNamespace(instruments=parts)
        bounded = SimpleNamespace(instruments=[SimpleNamespace(
            notes=[note for note, shared in zip(notes, item['shared'], strict=True) if shared])])
        probability = verifier.residual_model.probability(item['x'])
        expected = residual_keep(item, probability, verifier.residual_model.threshold)
        with patch.object(service.sf, 'read', return_value=(np.zeros(22050, dtype=np.float32), 22050)), \
                patch.object(service, 'note_features', return_value=features), \
                patch.object(service, 'decode_candidates', return_value=bounded):
            count = verifier.filter('cached-features-only.wav', {}, midi)
        assert count == int((~expected).sum()), item['id']
        for part, originals, mask in zip(parts, [notes[:split], notes[split:]],
                                         [expected[:split], expected[split:]], strict=True):
            assert [id(n) for n in part.notes] == [id(n) for n, keep in zip(originals, mask, strict=True) if keep], item['id']
        assert [vars(note) for note in notes] == attributes, item['id']
        total += len(notes)
        rejected += count
    result = {'passes': True, 'recordings': len(items), 'candidate_events': total,
              'total_rejected': rejected, 'model_sha256': service.RESIDUAL_SHA256,
              'checks': 'Runtime masks equal frozen evaluation; retained identity, source part and all note attributes preserved',
              'scope': 'Cached labeled dataset features only; no audio inference or score generation'}
    write(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.directory, args.output)
