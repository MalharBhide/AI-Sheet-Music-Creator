"""Verify the original small Oxford MIDI-test archive, without audio inference."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from prepare_robust_training_stems import digest, preserve

BASE = "https://www.robots.ox.ac.uk/~vgg/research/sighttosound/"
ARCHIVE = BASE + "resources/MIDItest.zip"
LICENSE = BASE + "resources/license_miditest.txt"
MAX_BYTES = 128 * 1024 * 1024


def safe_members(archive):
    seen = set()
    result = []
    for row in archive.infolist():
        path = PurePosixPath(row.filename)
        if (
            not row.filename
            or "\\" in row.filename
            or path.is_absolute()
            or ".." in path.parts
            or row.filename in seen
            or (row.external_attr >> 16) & 0o170000 == 0o120000
            or row.flag_bits & 1
            or row.file_size > MAX_BYTES
        ):
            raise ValueError("Unsafe original Oxford archive member")
        seen.add(row.filename)
        result.append(
            {"member": row.filename, "bytes": row.file_size, "crc32": f"{row.CRC:08x}"}
        )
    if sum(row["bytes"] for row in result) > 4 * MAX_BYTES:
        raise ValueError("Oversized original Oxford archive")
    bad = archive.testzip()
    if bad:
        raise ValueError("Corrupt original Oxford archive member: " + bad)
    return result


def acquire(output):
    output.mkdir(exist_ok=False)
    with urllib.request.urlopen(LICENSE, timeout=60) as response:
        license_bytes = response.read(16385)
    text = license_bytes.decode("utf-8")
    if len(license_bytes) > 16384 or any(
        name not in text
        for name in (
            "MIDI test Dataset",
            "Creative Commons Attribution 4.0 International",
            "Koepke",
            "ICASSP, 2020",
        )
    ):
        raise ValueError("Changed original Oxford MIDI-test license")
    (output / "LICENSE.publisher.txt").write_bytes(license_bytes)
    with urllib.request.urlopen(BASE, timeout=60) as response:
        page = response.read(128 * 1024 + 1)
    if len(page) > 128 * 1024 or b"MIDItest.zip" not in page:
        raise ValueError("Changed Oxford publisher page")
    (output / "publisher.html").write_bytes(page)
    preserve(
        output / "plan.json",
        {
            "archive_url": ARCHIVE,
            "license_url": LICENSE,
            "license_sha256": digest(output / "LICENSE.publisher.txt"),
            "publisher_sha256": digest(output / "publisher.html"),
            "producer_sha256": digest(Path(__file__)),
            "no_audio_or_labels_observed": True,
            "no_fitting_or_inference": True,
            "maximum_archive_bytes": MAX_BYTES,
        },
    )
    temporary = output / "source.zip.partial"
    total = 0
    sha = hashlib.sha256()
    with (
        urllib.request.urlopen(ARCHIVE, timeout=60) as response,
        temporary.open("xb") as stream,
    ):
        expected = response.headers.get("Content-Length")
        provenance = {
            k: response.headers.get(k)
            for k in ("ETag", "Last-Modified", "Content-Length")
        }
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("Oversized Oxford download")
            stream.write(chunk)
            sha.update(chunk)
    if expected is not None and total != int(expected):
        raise ValueError("Incomplete Oxford original archive")
    with zipfile.ZipFile(temporary) as archive:
        members = safe_members(archive)
    temporary.rename(output / "source.zip")
    preserve(
        output / "source.json",
        {
            "archive_url": ARCHIVE,
            "bytes": total,
            "sha256": sha.hexdigest(),
            "http_provenance": provenance,
            "members": members,
            "all_members_crc_verified": True,
            "publisher_supplies_no_checksum": True,
            "license": "CC BY 4.0",
            "plan_sha256": digest(output / "plan.json"),
            "no_audio_or_labels_observed": True,
            "no_fitting_or_inference": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "bytes": total,
                "members": len(members),
                "sha256": sha.hexdigest(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    acquire(parser.parse_args().output)
