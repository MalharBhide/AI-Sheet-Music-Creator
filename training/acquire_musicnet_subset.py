"""Stream and verify original MusicNet; retain bounded work-disjoint piano excerpts."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import struct
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath

from prepare_robust_training_stems import digest, preserve

RECORD = "https://zenodo.org/api/records/5120004"
URL = "https://zenodo.org/records/5120004/files/musicnet.tar.gz"
SIZE = 11097394998
MD5 = "844764911fa0d5b97c97da944a057590"
METADATA_MD5 = "1caef62cee9c875235e62aac368b49d8"
STARTS = (15, 35, 55)
SECONDS = 20


class CheckedReader:
    def __init__(self, source):
        self.source = source
        self.count = 0
        self.md5 = hashlib.md5()
        self.sha = hashlib.sha256()
        self.progress = 0

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded MusicNet stream read")
        data = self.source.read(size)
        self.count += len(data)
        self.md5.update(data)
        self.sha.update(data)
        if self.count > SIZE:
            raise ValueError("Oversized original MusicNet stream")
        if self.count - self.progress >= 128 * 1024 * 1024:
            self.progress = self.count
            print(json.dumps({"compressed_bytes_read": self.count, "total": SIZE}), flush=True)
        return data

    def verify(self):
        if self.count != SIZE or self.md5.hexdigest() != MD5:
            raise ValueError("Incomplete or corrupt original MusicNet archive")


def exact(source, amount):
    data = source.read(amount)
    if len(data) != amount:
        raise ValueError("Truncated MusicNet member")
    return data


def discard(source, amount):
    while amount:
        block = exact(source, min(amount, 1024 * 1024))
        amount -= len(block)


def crop_wave(source, size):
    """Parse RIFF without loading a full recording; preserve original sample clocks."""
    header = exact(source, 12)
    if header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise ValueError("Unsupported MusicNet waveform container")
    if struct.unpack("<I", header[4:8])[0] + 8 != size:
        raise ValueError("Changed RIFF member size")
    position, fmt, result = 12, None, None
    while position < size:
        tag, length = struct.unpack("<4sI", exact(source, 8))
        position += 8
        if position + length + length % 2 > size:
            raise ValueError("RIFF chunk escapes member")
        if tag == b"fmt ":
            if fmt is not None or not 16 <= length <= 64:
                raise ValueError("Invalid or duplicate waveform format")
            fmt = exact(source, length)
            code, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if (
                code not in (1, 3)
                or channels not in (1, 2)
                or rate != 44100
                or bits not in (16, 24, 32)
                or (code == 3 and bits != 32)
                or align != channels * (bits // 8)
                or byte_rate != rate * align
            ):
                raise ValueError("Unsupported original MusicNet sample format")
        elif tag == b"data":
            if fmt is None or result is not None or length % align:
                raise ValueError("Invalid waveform sample clock")
            duration = length / byte_rate
            if duration < STARTS[-1] + SECONDS:
                raise ValueError("Selected original waveform too short")
            result, consumed = [], 0
            for start in STARTS:
                first, amount = start * byte_rate, SECONDS * byte_rate
                discard(source, first - consumed)
                raw = exact(source, amount)
                # Keep the publisher's PCM/float samples with a new RIFF header.
                chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
                chunks += b"data" + struct.pack("<I", len(raw)) + raw
                wave = b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
                result.append(
                    {
                        "start_sample": start * rate,
                        "end_sample": (start + SECONDS) * rate,
                        "wave": wave,
                    }
                )
                consumed = first + amount
            discard(source, length - consumed)
        else:
            discard(source, length)
        if length % 2:
            exact(source, 1)
        position += length + length % 2
    if result is None:
        raise ValueError("Missing MusicNet waveform samples")
    return result, {
        "sample_rate": rate,
        "channels": channels,
        "format_code": code,
        "bits": bits,
        "original_duration": duration,
    }


def select(rows):
    works = defaultdict(list)
    for row in rows:
        if row["ensemble"] == "Solo Piano" and float(row["seconds"]) >= 75:
            works[(row["composer"], row["catalog_name"], row["composition"])].append(row)
    result = []
    composers = sorted({key[0] for key in works})
    if composers != ["Bach", "Beethoven", "Schubert"]:
        raise ValueError("Changed original solo-piano metadata coverage")
    for composer in composers:
        keys = sorted(key for key in works if key[0] == composer)
        count = min(12, len(keys))
        chosen = [keys[(index * (len(keys) - 1)) // (count - 1)] for index in range(count)]
        training, validation = count * 2 // 3, max(1, count // 6)
        for index, key in enumerate(chosen):
            row = min(works[key], key=lambda r: int(r["id"]))
            group = (
                "train"
                if index < training
                else "validation"
                if index < training + validation
                else "reserved"
            )
            result.append({**row, "group": group, "work": list(key)})
    if len(result) != 32 or len({r["id"] for r in result}) != 32:
        raise ValueError("Changed predetermined MusicNet work selection")
    return result


def safe_member(member):
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in member.name
        or not (member.isdir() or member.isfile())
        or not 0 <= member.size <= 2 * 1024**3
    ):
        raise ValueError("Unsafe original MusicNet TAR member")


def acquire(metadata, output):
    if output.exists():
        raise ValueError("Preserve existing MusicNet acquisition")
    if hashlib.md5(metadata.read_bytes()).hexdigest() != METADATA_MD5:
        raise ValueError("Changed publisher MusicNet metadata")
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        publisher = json.load(response)
    audio = [f for f in publisher["files"] if f["key"] == "musicnet.tar.gz"]
    if (
        publisher["id"] != 5120004
        or publisher["metadata"]["license"]["id"] != "cc-by-4.0"
        or len(audio) != 1
        or audio[0]["size"] != SIZE
        or audio[0]["checksum"] != "md5:" + MD5
    ):
        raise ValueError("Changed original MusicNet publisher license or archive identity")
    selected = select(list(csv.DictReader(io.StringIO(metadata.read_text()))))
    rows = {r["id"]: r for r in selected}
    output.mkdir()
    (output / "audio").mkdir()
    (output / "labels").mkdir()
    preserve(output / "publisher.json", publisher)
    (output / "metadata.csv").write_bytes(metadata.read_bytes())
    preserve(
        output / "plan.json",
        {
            "producer_sha256": digest(Path(__file__)),
            "publisher_sha256": digest(output / "publisher.json"),
            "metadata_sha256": digest(metadata),
            "selected": selected,
            "selection": "Metadata-only evenly spaced works per composer (at most 12), one lowest-ID recording per work; first two-thirds train, next one-sixth validation, remainder reserved. Only original train_data/train_labels members retained; original test members never retained, no replacement if a selected ID is test-only.",
            "starts_seconds": list(STARTS),
            "excerpt_seconds": SECONDS,
            "scope": "Project work-disjoint subset, not the official benchmark. Does not assert performer independence or upstream detector pretraining independence. Publisher reports approximately 4% label error.",
            "reserved_decoder_inference_not_allowed": True,
            "no_user_audio_or_scores": True,
            "attribution": "MusicNet: John Thickstun, Zaid Harchaoui and Sham M. Kakade, University of Washington. CC BY 4.0; original recording and transcription provenance retained per metadata.",
        },
    )
    found_audio, found_labels, seen = {}, {}, set()
    with urllib.request.urlopen(URL, timeout=60) as remote:
        if remote.status != 200 or int(remote.headers["Content-Length"]) != SIZE:
            raise ValueError("Changed original MusicNet transfer")
        checked = CheckedReader(remote)
        with gzip.GzipFile(fileobj=checked) as uncompressed:
            with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
                for member in archive:
                    safe_member(member)
                    if member.name in seen:
                        raise ValueError("Duplicate MusicNet member identity")
                    seen.add(member.name)
                    match = re.fullmatch(
                        r"(?:\./)?musicnet/(train_data|train_labels)/(\d+)\.(wav|csv)", member.name
                    )
                    if not match or match[2] not in rows:
                        continue
                    category, identity, extension = match.groups()
                    source = archive.extractfile(member)
                    if category == "train_data" and extension == "wav":
                        crops, clock = crop_wave(source, member.size)
                        records = []
                        for index, crop in enumerate(crops):
                            path = output / "audio" / f"{identity}-{index}.wav"
                            path.write_bytes(crop.pop("wave"))
                            records.append(
                                {**crop, "audio": str(path), "audio_sha256": digest(path)}
                            )
                        found_audio[identity] = {
                            "clock": clock,
                            "excerpts": records,
                            "original_member": member.name,
                        }
                    elif category == "train_labels" and extension == "csv":
                        if member.size > 8 * 1024 * 1024:
                            raise ValueError("Oversized original note labels")
                        path = output / "labels" / f"{identity}.csv"
                        path.write_bytes(exact(source, member.size))
                        found_labels[identity] = {
                            "labels": str(path),
                            "labels_sha256": digest(path),
                            "original_label_member": member.name,
                        }
                    else:
                        raise ValueError("Changed MusicNet member semantics")
            while uncompressed.read(1024 * 1024):
                pass  # gzip CRC and footer are checked, including archive padding.
        while checked.read(1024 * 1024):
            pass
        checked.verify()
    if set(found_audio) != set(found_labels) or len(found_audio) < 20:
        raise ValueError("Incomplete predetermined MusicNet audio/label pairs")
    records = [
        {**rows[k], **found_audio[k], **found_labels[k]} for k in sorted(found_audio, key=int)
    ]
    preserve(
        output / "source.json",
        {
            "url": URL,
            "bytes": checked.count,
            "md5": checked.md5.hexdigest(),
            "sha256": checked.sha.hexdigest(),
            "complete_original_object_verified": True,
            "license": "CC BY 4.0",
            "publisher_record": RECORD,
            "no_full_archive_stored": True,
        },
    )
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "source_sha256": digest(output / "source.json"),
            "items": records,
            "excluded_selected_test_only_ids": sorted(set(rows) - set(found_audio)),
            "work_disjoint": True,
            "reserved_inference_not_performed": True,
            "no_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "complete_archive_verified": True,
                "recordings": len(records),
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
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.metadata, args.output)
