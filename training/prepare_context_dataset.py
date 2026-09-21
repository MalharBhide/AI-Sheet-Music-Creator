"""Assemble reproducible context-feature manifests from the corrected split."""

import argparse
import copy
import json
from pathlib import Path

from app.services.note_evidence import CONTEXT_VERSION


def prepare(directory):
    base = json.loads((directory / 'note-verifier-v3-data/note-verifier-split.json').read_text())
    base['version'] = CONTEXT_VERSION
    for items in base['tracks'].values():
        for item in items:
            item.update(context_features=True, candidate_decoder='bounded-accompaniment-v1')
    expanded = copy.deepcopy(base)
    for item in json.loads((directory / 'context-piano/manifest.json').read_text())['items']:
        expanded['tracks'][item['group']].append(item)
    for item in json.loads((directory / 'note-verifier-training-stems/manifest.json').read_text())['items']:
        expanded['tracks']['train'].append({**item, 'candidate_decoder': 'bounded-accompaniment-v1'})
    expanded['augmentation'] = '64 new original dense/quiet piano phrases train; 16 disjoint seeds validation; 40 source-separated examples from training performers 00–03 only'
    for name, manifest in [('note-verifier-context-data', base), ('note-verifier-context-expanded', expanded)]:
        output = directory / name
        output.mkdir(exist_ok=False)
        (output / 'note-verifier-split.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
