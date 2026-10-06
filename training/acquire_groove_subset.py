"""Verify the complete official GMD object, then retain bounded performer-disjoint audio."""

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from prepare_robust_training_stems import digest, preserve

URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"
SIZE = 5111599714
SHA = "21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9"
LICENSE = "https://magenta.tensorflow.org/datasets/groove"
MAX_MEMBER = 12 * 1024 * 1024


def identity():
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as source:
        if source.status != 200 or int(source.headers["Content-Length"]) != SIZE:
            raise ValueError("Changed original Groove archive size")
        generation = source.headers["x-goog-generation"]
        etag = source.headers["ETag"]
        if not generation.isdigit() or not re.fullmatch(r'"[a-f0-9]{32}"', etag):
            raise ValueError("Missing immutable original Groove object identity")
    return {"url": URL, "generation": generation, "etag": etag, "bytes": SIZE}


def verified_source(witness):
    remote = identity()
    expected = {
        **remote,
        "sha256": SHA,
        "license": "CC BY 4.0",
        "publisher_license_url": LICENSE,
        "producer_sha256": digest(Path(__file__)),
        "complete_original_object_verified": True,
        "no_full_archive_stored": True,
    }
    if witness.exists():
        if json.loads(witness.read_text()) != expected:
            raise ValueError("Changed complete Groove checksum witness")
        return expected
    hasher, count = hashlib.sha256(), 0
    address = URL + "?generation=" + remote["generation"]
    with urllib.request.urlopen(address, timeout=60) as source:
        if (
            source.status != 200
            or source.headers["x-goog-generation"] != remote["generation"]
            or source.headers["ETag"] != remote["etag"]
            or int(source.headers["Content-Length"]) != SIZE
        ):
            raise ValueError("Changed immutable Groove checksum stream")
        while block := source.read(1024 * 1024):
            hasher.update(block)
            count += len(block)
            if count % (128 * 1024 * 1024) == 0:
                print(json.dumps({"verified_bytes_read": count, "total_bytes": SIZE}), flush=True)
    if count != SIZE or hasher.hexdigest() != SHA:
        raise ValueError("Incomplete or corrupt publisher Groove archive")
    preserve(witness, expected)
    return expected


class RangeReader(io.RawIOBase):
    """Bounded reads from the immutable object whose complete SHA has been verified."""

    def __init__(self, source):
        if (
            source["sha256"] != SHA
            or source["bytes"] != SIZE
            or not source["complete_original_object_verified"]
        ):
            raise ValueError("Groove range access requires complete publisher checksum witness")
        self.source, self.position = source, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            0 if whence == 0 else self.position if whence == 1 else SIZE if whence == 2 else None
        )
        if position is None or not 0 <= position + offset <= SIZE:
            raise ValueError("Invalid Groove object range")
        self.position = position + offset
        return self.position

    def read(self, amount=-1):
        amount = SIZE - self.position if amount < 0 else min(amount, SIZE - self.position)
        if amount > MAX_MEMBER:
            raise ValueError("Unbounded Groove range read")
        if amount == 0:
            return b""
        first, last = self.position, self.position + amount - 1
        request = urllib.request.Request(
            URL + "?generation=" + self.source["generation"],
            headers={"Range": f"bytes={first}-{last}", "If-Match": self.source["etag"]},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers["Content-Range"] != f"bytes {first}-{last}/{SIZE}"
                or response.headers["x-goog-generation"] != self.source["generation"]
                or response.headers["ETag"] != self.source["etag"]
                or int(response.headers["Content-Length"]) != amount
            ):
                raise ValueError("Changed or ignored immutable Groove range request")
            data = response.read(amount + 1)
        if len(data) != amount:
            raise ValueError("Incomplete Groove object range")
        self.position += amount
        return data


def safe_member(member):
    path = PurePosixPath(member.filename)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in member.filename
        or member.file_size > MAX_MEMBER
        or member.compress_size > MAX_MEMBER
        or ((member.external_attr >> 16) & 0o170000) == 0o120000
        or member.flag_bits & 1
    ):
        raise ValueError("Unsafe or oversized selected Groove ZIP member")


def select(rows):
    performers = sorted({r["drummer"] for r in rows})
    if len(performers) != 10:
        raise ValueError("Changed original Groove performer coverage")
    result = []
    candidates = {
        performer: sorted(
            [
                r
                for r in rows
                if r["drummer"] == performer
                and r["audio_filename"]
                and r["split"] in ("train", "validation")
                and 8 <= float(r["duration"]) <= 40
            ],
            key=lambda r: (r["style"], r["id"], r["audio_filename"]),
        )
        for performer in performers
    }
    eligible = [performer for performer in performers if candidates[performer]]
    if len(eligible) != 9:
        raise ValueError("Changed bounded non-test Groove performer coverage")
    for index, performer in enumerate(eligible):
        choices = sorted(
            [
                r
                for r in rows
                if r["drummer"] == performer
                and r["audio_filename"]
                and r["split"] in ("train", "validation")
                and 8 <= float(r["duration"]) <= 40
            ],
            key=lambda r: (r["style"], r["id"], r["audio_filename"]),
        )
        # Human rhythm variety, bounded clip lengths. Metadata alone selects;
        # predictions, acoustic analysis and note-reference outcomes never do.
        styles, chosen = set(), []
        for row in choices:
            style = row["style"].split("/")[0]
            if style in styles:
                continue
            styles.add(style)
            chosen.append(row)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            chosen.extend(r for r in choices if r not in chosen)
            chosen = chosen[:2]
        group = "train" if index < 5 else "validation" if index < 7 else "reserved"
        result.extend({**row, "group": group} for row in chosen)
    return result


def acquire(witness, output):
    source = verified_source(witness)
    if output.exists():
        raise ValueError("Preserve existing Groove acquisition")
    with zipfile.ZipFile(RangeReader(source)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate original Groove ZIP identities")
        metadata = archive.getinfo("groove/info.csv")
        safe_member(metadata)
        payload = archive.read(metadata)
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        selected = select(rows)
        output.mkdir()
        (output / "audio").mkdir()
        (output / "midi").mkdir()
        (output / "info.csv").write_bytes(payload)
        preserve(output / "source.json", source)
        preserve(
            output / "plan.json",
            {
                "source_sha256": digest(output / "source.json"),
                "producer_sha256": digest(Path(__file__)),
                "metadata_sha256": digest(output / "info.csv"),
                "selection": "Original train/validation only, duration 8–40 seconds; at most two clips per eligible performer preferring distinct primary styles; nine eligible performers sorted first five train, next two validation, last two reserved. All original test sequences untouched.",
                "scope": "Project-specific performer split, not the official GMD benchmark. Electronic TD-11 drum audio is a nuisance source for piano transcription; drum MIDI numbers are not piano pitch labels.",
                "no_decoder_inference": True,
                "no_user_audio_or_scores": True,
                "reserved_inference_policy": "After a frozen validation winner passes all consumed regression recordings; never fitting or selection",
            },
        )
        records = []
        for index, row in enumerate(selected):
            record = {**row, "id": f"groove-source-{index:02d}", "publisher_id": row["id"]}
            for key, directory, suffix in (
                ("audio_filename", "audio", ".wav"),
                ("midi_filename", "midi", ".mid"),
            ):
                name = "groove/" + row[key]
                member = archive.getinfo(name)
                safe_member(member)
                path = output / directory / (record["id"] + suffix)
                path.write_bytes(archive.read(member))  # zipfile verifies the original CRC.
                record[directory] = str(path)
                record[directory + "_sha256"] = digest(path)
            records.append(record)
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "performer_disjoint": True,
            "reserved_inference_not_performed": True,
            "no_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "source_recordings": len(records),
                "groups": {
                    g: sum(r["group"] == g for r in records)
                    for g in ("train", "validation", "reserved")
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.witness, args.output)
