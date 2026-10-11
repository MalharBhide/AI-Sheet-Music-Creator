"""Reuse the checked boundary engine with training-only protected targets."""

import argparse
from pathlib import Path

from protected_boundary_targets_v37 import protected_labels
from train_chord_boundary_v36 import fit

VERSION = "protected-independent-boundaries-v37-v1"
PROFILES = (
    {"name": "protected-boundaries", "width": 16, "seed": 282001, "keep_weight": 8},
    {"name": "protected-holds", "width": 24, "seed": 282002, "keep_weight": 16},
)
POLICY = ("Training targets abstain from corrections that lose reference-pitch support, "
          "break a protected attack/hold, worsen the fixed reference-boundary error, "
          "distort an observed repeated-key/chord attack relation or increase unsupported "
          "sustain. Reference labels remain supervision only. Inference and all validation/"
          "release gates are identical to V36; no test labels or regressed candidate is reused.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "piano", "output"):
        parser.add_argument(name, type=Path)
    fit(**vars(parser.parse_args()), labeler=protected_labels, profiles=PROFILES, version=VERSION,
        extra_sources=("protected_boundary_targets_v37.py", "train_protected_boundaries_v37.py"),
        target_policy=POLICY)
