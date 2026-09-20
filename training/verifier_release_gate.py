"""Reject accompaniment candidates that trade current accuracy for fewer notes."""

import argparse
import json
from pathlib import Path


def decision(report):
    reasons = []
    for row in report['per_recording']:
        trained, original = row['trained'], row['baseline']
        if trained['f1'] < original['f1'] or trained['recall'] < original['recall'] - .02:
            reasons.append(f"{row['id']}: individual recording regressed against original detector")
    for corpus, row in report['aggregate'].items():
        for comparator in ('baseline', 'previous_v2'):
            for metric in ('precision', 'recall', 'f1'):
                if row['trained'][metric] < row[comparator][metric]:
                    reasons.append(f'{corpus}: {metric} regressed against {comparator}')
    if not any(row['trained']['false_positives'] < row['previous_v2']['false_positives']
               for row in report['aggregate'].values()):
        reasons.append('No corpus reduces false notes compared with the deployed model')
    return {'promoted': not reasons, 'reasons': reasons,
            'scope': 'Regression evidence only; passing does not guarantee quality on every upload'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(decision(json.loads(args.report.read_text())), indent=2))
