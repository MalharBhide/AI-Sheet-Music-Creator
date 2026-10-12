"""Inventory pinned PDMX v9 symbolic archives; no extraction or training clearance."""

import argparse
import collections
import csv
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

FILES = {
    "mid": (214395208, "d920a21b2fcd99a56d9c381b39debbb2", ".mid", 254035),
    "metadata": (159444765, "5bc79445090dd2fe5e96cffa77a3461c", ".json", 254077),
}
CATALOG = (225399738, "30392ccf38bb63ce70e7afae70f9c88c")


def verify_file(path, size, expected_md5):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if path.stat().st_size != size or md5.hexdigest() != expected_md5:
        raise ValueError("Publisher size/checksum mismatch")
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def safe_path(name, root):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or not path.parts or path.parts[0] != root):
        raise ValueError("Unsafe or unexpected symbolic path")
    return str(path)


def catalog_path(value, root):
    # Publisher uses NA for exactly the 42 invalid/missing MIDI derivatives.
    return None if value in ("", "NA") else safe_path(value, root)


def inventory(path, root, suffix, expected_count):
    names, directories, total = set(), set(), 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = safe_path(member.name, root)
            if name in names or name in directories:
                raise ValueError("Duplicate symbolic archive member")
            if member.isdir():
                directories.add(name)
            elif member.isfile() and name.endswith(suffix) and 0 < member.size <= 64 * 1024 * 1024:
                names.add(name)
                total += member.size
            else:
                raise ValueError("Link, special, oversized or unexpected archive member")
    if len(names) != expected_count:
        raise ValueError("Publisher symbolic member count changed")
    return names, {"regular_files": len(names), "uncompressed_bytes": total,
                   "directories": len(directories), "safe_unique_paths": True}


def audit(catalog, midi, metadata):
    report = {"version": "pdmx-v9-symbolic-integrity-v1",
              "publisher": "https://zenodo.org/records/15571083",
              "catalog": verify_file(catalog, *CATALOG), "archives": {},
              "training_cleared": False, "extracted_members": 0,
              "audio_rendered_or_inferred": False, "user_audio_or_scores_processed": False}
    inventories = {}
    for root, path in (("mid", midi), ("metadata", metadata)):
        size, md5, suffix, count = FILES[root]
        proof = verify_file(path, size, md5)
        inventories[root], counts = inventory(path, root, suffix, count)
        report["archives"][root] = {**proof, **counts}
    counts = collections.Counter()
    references = {root: set() for root in inventories}
    with catalog.open(newline="") as reader:
        for row in csv.DictReader(reader):
            counts["rows"] += 1
            clean = row["subset:no_license_conflict"] == "True" and row["subset:all_valid"] == "True"
            counts["conflict_free_valid_rows"] += int(clean)
            for root, members in inventories.items():
                name = catalog_path(row[root], root)
                if name is None:
                    counts[f"{root}_empty_references"] += 1
                    missing = True
                else:
                    references[root].add(name)
                    missing = name not in members
                counts[f"{root}_missing_references"] += int(missing)
                counts[f"{root}_clean_valid_missing_references"] += int(missing and clean)
    if counts["rows"] != 254077 or counts["conflict_free_valid_rows"] != 222820:
        raise ValueError("Publisher catalog population changed")
    for root, members in inventories.items():
        counts[f"{root}_unreferenced_archive_members"] = len(members - references[root])
        if counts[f"{root}_clean_valid_missing_references"]:
            raise ValueError("A clean valid row is missing its symbolic file")
    report["catalog_links"] = dict(counts)
    report["limitations"] = [
        "Integrity and catalog linkage only; no per-score rights or work identity clearance.",
        "Public-domain labels and program zero do not alone establish piano intent or rights.",
        "Clock, controller, same-key overlap and source-group checks required before rendering.",
        "Any future sampled audio supplies synthetic commands, not human release or pedal labels.",
    ]
    report["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.catalog, args.midi, args.metadata)
    with args.output.open("x") as writer:
        json.dump(result, writer, indent=2)
