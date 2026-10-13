"""Acquire only pinned CC0 piano commands/metadata, never remote audio or code."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

COMMIT = "f4b3d0582c7a1ecb725634383ccd41379d2587a2"
BASE = "https://raw.githubusercontent.com/carf-coder/keynata-commons/" + COMMIT + "/"
VERSION = "keynata-piano-symbolic-source-assessment-v55-v1"


def blob_digest(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data,
                        usedforsecurity=False).hexdigest()


def verify(data, entry):
    if (entry["type"] != "blob" or entry["mode"] != "100644"
            or entry["size"] != len(data) or blob_digest(data) != entry["sha"]):
        raise ValueError("Changed pinned publisher blob")
    return hashlib.sha256(data).hexdigest()


def piano_rows(index):
    tracks = index["tracks"]
    if (index["license"]["generated_content"] != "CC0-1.0"
            or index["n_tracks"] != len(tracks) or len(tracks) != 203
            or len({t["track_id"] for t in tracks}) != len(tracks)
            or dict(Counter(t["category"] for t in tracks)) != index["tracks_by_category"]):
        raise ValueError("Changed publisher inventory or license")
    rows = [t for t in tracks if t["category"] == "classical_piano"]
    if len(rows) != 19:
        raise ValueError("Incomplete fixed piano inventory")
    for row in rows:
        name = row["track_id"]
        if (not re.fullmatch(r"classical_piano_[0-9]{4,6}", name)
                or row["instrumentation"] != ["piano"] or not row["screening_passed"]
                or row["versions"] != ["main"] or row["files"] != [name+".mid", name+".mp3"]):
            raise ValueError("Ambiguous publisher piano identity")
    return rows


def acquire(root):
    tree = json.loads((root/"github-tree.json").read_text())
    if tree["sha"] != COMMIT or tree["truncated"]:
        raise ValueError("Changed or truncated publisher tree")
    entries = {e["path"]: e for e in tree["tree"]}
    if len(entries) != len(tree["tree"]):
        raise ValueError("Duplicated publisher tree path")
    index_data = (root/"index.json").read_bytes()
    license_data = (root/"LICENSE.txt").read_bytes()
    verify(index_data, entries["release/index.json"])
    verify(license_data, entries["release/LICENSE.txt"])
    index = json.loads(index_data)
    rows = piano_rows(index)
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = [f"release/tracks/{r['track_id']}/{name}"
            for r in rows for name in (r["track_id"]+".mid", "metadata.json")]
    plan = {"version": VERSION, "commit": COMMIT, "paths": jobs,
        "license_sha256": hashlib.sha256(license_data).hexdigest(),
        "index_sha256": hashlib.sha256(index_data).hexdigest(),
        "tree_sha256": hashlib.sha256((root/"github-tree.json").read_bytes()).hexdigest(),
        "producer_sha256": producer, "no_audio_acquisition": True,
        "source_groups_training_only_if_later_admitted": True, "new_weights_deployed": False}
    target = root/"acquisition-plan.json"
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError("Preserve changed source acquisition")
    else:
        target.write_text(json.dumps(plan, indent=2)+"\n")
    records = []
    for path in jobs:
        entry = entries[path]
        if not 0 < entry["size"] <= 1_000_000:
            raise ValueError("Unexpected symbolic source size")
        destination = root/path
        if not destination.exists():
            with urlopen(BASE+path, timeout=30) as response:
                data = response.read(1_000_001)
            verify(data, entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        data = destination.read_bytes()
        records.append({"path": path, "bytes": len(data), "git_blob_sha1": entry["sha"],
                        "sha256": verify(data, entry)})
        print(json.dumps(records[-1]), flush=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != producer:
        raise ValueError("Changed symbolic source producer during acquisition")
    manifest = {"version": VERSION, "commit": COMMIT, "files": records,
        "plan_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "no_audio_acquisition": True, "no_model_training_or_inference": True,
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    target = root/"acquisition-manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Preserve completed source acquisition")
    else:
        target.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    acquire(**vars(parser.parse_args()))
