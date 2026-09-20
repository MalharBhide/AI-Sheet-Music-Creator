"""Catch gross clock/label failures before fitting an accompaniment model."""

from collections import defaultdict

import numpy as np


def audit(training, validation):
    groups = defaultdict(list)
    for split, items in [('train', training), ('validation', validation)]:
        for item in items:
            if item['corpus'] == 'vienna-piano':
                groups[(split, item['id'].rsplit('_p', 1)[0])].append(item)
    rows = []
    for (split, work), items in sorted(groups.items()):
        matched = sum(int(np.sum(item['y'])) for item in items)
        predicted = sum(len(item['events']) for item in items)
        reference = sum(len(item['reference']) for item in items)
        f1 = 2 * matched / max(1, predicted + reference)
        rows.append({'split': split, 'work': work, 'recordings': len(items), 'detector_f1': f1})
    # This is a gross alignment check on this known corpus, not an accuracy goal
    # for arbitrary instruments or a substitute for checking its audio clocks.
    if any(row['detector_f1'] < .5 for row in rows):
        raise ValueError(f'Possible Vienna label/clock failure; inspect before training: {rows}')
    return rows
