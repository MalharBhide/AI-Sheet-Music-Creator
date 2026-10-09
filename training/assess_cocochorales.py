"""Verify a new licensed source using metadata only; do not decode audio or fit."""

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import yaml
from prepare_robust_training_stems import digest, preserve

PUBLISHER = "https://magenta.withgoogle.com/datasets/cocochorales"
BASE = (
    "https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/"
)
FORMAT = (
    "https://raw.githubusercontent.com/lukewys/chamber-ensemble-generator/master/data_format.md"
)
ATTRIBUTION = "Yusong Wu, Josh Gardner, Ethan Manilow, Ian Simon, Curtis Hawthorne, Jesse Engel. The Chamber Ensemble Generator: Limitless High-Quality MIR Data via Generative Modeling (2022)."


def fetch(url, limit):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Source exceeds metadata download bound")
        if response.headers.get("Content-Length") and len(data) != int(
            response.headers["Content-Length"]
        ):
            raise ValueError("Incomplete source transfer")
        return data


def checksum_list(data):
    result = {}
    for line in data.decode().splitlines():
        match = re.fullmatch(r"([a-f0-9]{32})\s+\./([\w/.-]+)", line)
        if not match:
            raise ValueError("Invalid publisher checksum line")
        checksum, name = match.groups()
        if name in result or ".." in PurePosixPath(name).parts:
            raise ValueError("Duplicate or unsafe publisher checksum identity")
        result[name] = checksum
    return result


def read_metadata(data):
    rows, seen, total = [], set(), 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe metadata member")
            if member.isdir():
                continue
            if (
                not member.isfile()
                or member.size > 1024 * 1024
                or path.suffix not in (".yaml", ".yml")
            ):
                raise ValueError("Unexpected metadata member")
            total += member.size
            if total > 64 * 1024 * 1024 or member.name in seen or len(rows) >= 5000:
                raise ValueError("Oversized or duplicate metadata")
            seen.add(member.name)
            raw = archive.extractfile(member).read()
            if len(raw) != member.size:
                raise ValueError("Truncated metadata member")
            value = yaml.safe_load(raw)
            if not isinstance(value, dict) or not {
                "midi_file",
                "ensemble",
                "instrument_name",
            }.issubset(value):
                raise ValueError("Missing source grouping or ensemble metadata")
            source = str(value["midi_file"])
            if not re.fullmatch(r"\d+\.mid", source):
                raise ValueError("Invalid original MIDI identity")
            rows.append(
                {
                    "member": member.name,
                    "original_midi": source,
                    "ensemble": value["ensemble"],
                    "instruments": value["instrument_name"],
                    "member_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    if not rows:
        raise ValueError("Empty metadata package")
    return rows


def assess(output):
    output.mkdir(exist_ok=False)
    plan = {
        "publisher": PUBLISHER,
        "base": BASE,
        "format": FORMAT,
        "archives": ["metadata/train/1.tar.bz2", "metadata/valid/1.tar.bz2"],
        "producer_sha256": digest(Path(__file__)),
        "no_audio_midi_or_note_labels": True,
        "no_fitting_selection_inference_or_user_audio": True,
        "metadata_grouping_policy": "All recordings from the same original midi_file must share a split across ensembles; official numeric recording splits alone are insufficient to prove independence.",
    }
    preserve(output / "plan.json", plan)
    publisher = fetch(PUBLISHER, 2 * 1024 * 1024)
    if b"CC-BY 4.0" not in publisher or b"Yusong Wu" not in publisher:
        raise ValueError("Publisher no longer declares expected dataset rights")
    docs = fetch(FORMAT, 2 * 1024 * 1024)
    sums = fetch(BASE + "cocochorales_md5s.txt", 2 * 1024 * 1024)
    checksums = checksum_list(sums)
    for name, data in (
        ("publisher.html", publisher),
        ("format.md", docs),
        ("publisher-md5.txt", sums),
    ):
        (output / name).write_bytes(data)
    archives = []
    for name in plan["archives"]:
        data = fetch(BASE + name, 32 * 1024 * 1024)
        if hashlib.md5(data).hexdigest() != checksums[name]:
            raise ValueError("Publisher metadata checksum mismatch")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows = read_metadata(data)
        archives.append(
            {
                "archive": name,
                "md5": checksums[name],
                "sha256": digest(target),
                "bytes": len(data),
                "records": rows,
            }
        )
    overlap = sorted(
        {r["original_midi"] for r in archives[0]["records"]}
        & {r["original_midi"] for r in archives[1]["records"]}
    )
    result = {
        "license": "CC BY 4.0",
        "attribution": ATTRIBUTION,
        "plan_sha256": digest(output / "plan.json"),
        "archives": archives,
        "observed_original_midi_overlap": overlap,
        "audio_acquired": False,
        "trained": False,
        "suitability": "Synthetic chamber sources for harmonic false-note and separation-leakage supervision. Contains no piano; cannot establish real-piano improvement. Need complete verified audio/label acquisition, source-group splits, physical observations and independent real-piano gates before fitting or release.",
    }
    preserve(output / "assessment.json", result)
    print(
        json.dumps(
            {
                "completed": True,
                "records": [len(a["records"]) for a in archives],
                "source_overlap": len(overlap),
                "trained": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    assess(parser.parse_args().output)
