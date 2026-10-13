"""Additional learned removal permissions cannot restore V43 rejected notes."""

import numpy as np
from left_precision_v44 import evaluate as replacement_evaluate
from left_precision_v45 import pitch_evaluate as replacement_pitch_evaluate

VERSION = "monotone-weak-pitch-removal-v46-v1"


def permissions(probability, baseline_accepted):
    probability, accepted = np.asarray(probability), np.asarray(baseline_accepted)
    if (accepted.ndim != 1 or probability.shape != (len(accepted), 2)
            or accepted.dtype != np.bool_ or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Invalid additional removal permissions")
    result = probability.copy()
    result[~accepted] = [0., 1.]
    return result


def locked(items, deployed, probabilities):
    result = []
    for original, baseline, probability in zip(items, deployed, probabilities, strict=True):
        if original["id"] != baseline["id"]:
            raise ValueError("Changed baseline removal identity")
        events, accepted = original["events"], np.asarray(baseline["baseline_accepted"])
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= original["duration"]-2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        if accepted.shape != eligible.shape or accepted.dtype != np.bool_ or np.any(~accepted & ~eligible):
            raise ValueError("Baseline rejected an ineligible note")
        result.append(permissions(probability, accepted))
    return result


def evaluate(items, deployed, probabilities, threshold, memo=None):
    report = replacement_evaluate(items, deployed, locked(items, deployed, probabilities), threshold, memo)
    if any(row["recovered_notes"] != 0 for row in report["per_recording"]):
        raise ValueError("Additional verifier restored a baseline rejection")
    return report


def pitch_evaluate(items, deployed, probabilities, threshold):
    report = replacement_pitch_evaluate(items, deployed, locked(items, deployed, probabilities), threshold)
    if any(row["recovered_notes"] != 0 for row in report["per_recording"]):
        raise ValueError("Additional pitch verifier restored a baseline rejection")
    return report
