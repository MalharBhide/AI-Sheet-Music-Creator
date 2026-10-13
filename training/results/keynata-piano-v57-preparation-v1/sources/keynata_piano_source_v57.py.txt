"""Admit only the five previously audited CC0 command sources to fitting."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

from acquire_keynata_piano_v55 import verify
from audit_keynata_piano_v55_v2 import audit
from prepare_robust_training_stems import digest

AUDIT_SHA256 = "5efa02b0294fc49e01b8cf12f51e9ede13cba8dc8e33c0c05cb2a188e814d8a5"
WINDOWS = {"classical_piano_4014": 1, "classical_piano_5702": 5,
           "classical_piano_5777": 6, "classical_piano_5901": 6,
           "classical_piano_6134": 6}


def sources(root):
    cached = root / "symbolic-audit-v2.json"
    if digest(cached) != AUDIT_SHA256:
        raise ValueError("Changed fixed pre-training symbolic audit")
    # Revalidate source bytes and every command, rather than trusting a manifest
    # flag. The original assessment remains immutable and has no audio access.
    with tempfile.TemporaryDirectory(prefix="keynata-v57-audit-") as temporary:
        output = Path(temporary) / "audit.json"
        with contextlib.redirect_stdout(io.StringIO()):
            audit(root, output)
        checked = json.loads(output.read_text())
    if checked != json.loads(cached.read_text()):
        raise ValueError("Changed audited source population or commands")
    tree = json.loads((root / "github-tree.json").read_text())
    entries = {entry["path"]: entry for entry in tree["tree"]}
    for local, remote in (("index.json", "release/index.json"),
                          ("LICENSE.txt", "release/LICENSE.txt")):
        verify((root / local).read_bytes(), entries[remote])
    admitted = [row for row in checked["sources"] if row["eligible_unambiguous_commands"]]
    if len(admitted) != 5 or {row["id"] for row in admitted} != set(WINDOWS):
        raise ValueError("Changed fixed admitted source cohort")
    return [{"id": row["id"], "group": "train", "source_group": row["source_group"],
             "midi": str(root / "release/tracks" / row["id"] / (row["id"] + ".mid")),
             "midi_sha256": row["source_midi_sha256"],
             "metadata_sha256": row["metadata_sha256"],
             "source_seconds": row["last_key_release_seconds"],
             "command_array_sha256": row["command_array_sha256"],
             "license": row["license"], "human_performance": False}
            for row in admitted]
