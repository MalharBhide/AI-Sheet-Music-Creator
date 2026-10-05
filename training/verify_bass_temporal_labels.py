"""Verify the new label loader keeps every existing protection exactly."""

import argparse
import json
from pathlib import Path

import numpy as np
from bass_temporal_data import baseline_items
from bass_residual_coverage_data import protect
from bass_residual_v11_data import annotate
from prepare_robust_training_stems import digest, preserve


def verify(cache, destination):
    items = baseline_items(cache, ('train', 'validation', 'test'))
    for item in items:
        historical = protect(annotate({**item, 'x': item['base_x'],
                     'context': np.zeros(len(item['events'])), 'guardian': np.zeros(len(item['events']))}))
        for key in ('y', 'mask', 'promoted', 'baseline_keep', 'keep'):
            np.testing.assert_array_equal(item[key], historical[key], err_msg=item['id'])
        assert item['protected_offset_assignments'] == historical['protected_offset_assignments'], item['id']
        assert item['partial_pitch_support_promoted'] == historical['partial_pitch_support_promoted'], item['id']
    result = {'passes': True, 'recordings': len(items), 'v14_retained_events': sum(len(i['events']) for i in items),
              'source_manifest_sha256': digest(cache / 'manifest.json'), 'script_sha256': digest(Path(__file__)),
              'scope': 'Exact attack/hold/partial-pitch KEEP supervision parity with historical labels; zero confidence placeholders affect only unused historical features. No fitting, model selection or test tuning.'}
    preserve(destination, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    verify(args.cache, args.destination)
