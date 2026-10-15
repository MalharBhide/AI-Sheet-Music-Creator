"""Resume the released piano verifier without resetting weights or normalization.

This is initialization and storage preflight, not a release qualification.
Fitting, validation, consumed regressions and fresh evaluation remain mandatory.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

APPROVED_SHA256 = "38899dd8dbbefafb3a2d5d2f210dc2e6db66087ffc6ee95fcae2635f5d9bbfda"
LOCAL_LIMIT_BYTES = 25_000_000_000


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def storage_preflight(root, incoming_bytes=0, minimum_free_bytes=1_000_000_000):
    """Count all training files, refusing links and insufficient reserved space.

    Run before each bounded acquisition/preparation stage, reserving that stage's
    maximum simultaneous input, temporary and output bytes, not just download size.
    The check is not a filesystem quota or protection against concurrent writers.
    """
    root = Path(root)
    if (root.is_symlink() or not root.is_dir() or type(incoming_bytes) is not int
            or incoming_bytes < 0 or type(minimum_free_bytes) is not int
            or minimum_free_bytes < 0):
        raise ValueError("Invalid training storage reservation")
    used = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Training storage contains a symlink; audit its target first")
        if path.is_file():
            info = path.stat()
            used += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
    free = shutil.disk_usage(root).free
    if used + incoming_bytes > LOCAL_LIMIT_BYTES:
        raise ValueError("Training stage would exceed the 25 GB local limit")
    if free - incoming_bytes < minimum_free_bytes:
        raise ValueError("Training stage would leave insufficient laptop disk space")
    return {"used_bytes": used, "reserved_incoming_bytes": incoming_bytes,
            "limit_bytes": LOCAL_LIMIT_BYTES, "free_bytes": free}


def load_approved(production):
    """Copy the deployed, hash-pinned network with its original fitted normalizer."""
    from app.services.left_presence import PresenceModel

    production = Path(production)
    path = production / "backend/app/assets/left-presence-v1.npz"
    release = json.loads((production / "training/results/left-precision-v43-v1/release-checks.json").read_text())
    if (release["status"] != "verified" or release["new_weights_deployed"] is not True
            or release["no_user_audio_or_scores"] is not True
            or release["asset_sha256"] != APPROVED_SHA256 or sha256(path) != APPROVED_SHA256):
        raise ValueError("Only the approved deployed model may initialize fine-tuning")
    with np.load(path, allow_pickle=False) as saved:
        approved = PresenceModel(saved)
    # Do not recompute normalization on a different dataset: that changes every
    # inherited prediction before the first optimization step.
    for array in approved.normalizer:
        array.setflags(write=False)
    return approved


def verify_retained_checkpoint(approved, checkpoint, expected_sha256):
    """Prove the retained fitting checkpoint and deployed initialization are equal."""
    import torch

    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Changed retained approved checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["version"] != "calibrated-frame-support-v43-v1"
            or saved["width"] != 12 or saved["epoch"] != 50):
        raise ValueError("Unapproved retained checkpoint metadata")
    state = approved.network.state_dict()
    if set(saved["state_dict"]) != set(state):
        raise ValueError("Retained checkpoint architecture differs")
    for key, value in state.items():
        other = saved["state_dict"][key]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            raise ValueError("Retained checkpoint weights differ from deployed model")
    for key, value in zip(("mean", "scale"), approved.normalizer, strict=True):
        other = saved[key].numpy()
        if value.dtype != other.dtype or not np.array_equal(value, other):
            raise ValueError("Retained checkpoint normalization differs")
    return {"approved_asset_sha256": APPROVED_SHA256,
            "retained_checkpoint_sha256": expected_sha256,
            "all_weights_and_normalization_identical": True,
            "architecture_changed": False, "new_weights_deployed": False}
