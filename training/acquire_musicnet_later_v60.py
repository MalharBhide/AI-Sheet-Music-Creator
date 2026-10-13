"""Stream the pinned original object; retain only audited later fitting passages."""

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import re
import struct
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from audit_musicnet_expansion_v59 import MD5, RATE, SECONDS, SHA256, SIZE, audit, digest

VERSION = "musicnet-later-original-fitting-source-v60-v1"
URL = "https://zenodo.org/records/5120004/files/musicnet.tar.gz"
RECORD = "https://zenodo.org/api/records/5120004"


def preserve(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def same_write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Changed partial original source bytes")
    else:
        path.write_bytes(data)


class CheckedReader:
    def __init__(self, source):
        self.source, self.count, self.progress = source, 0, 0
        self.md5, self.sha = hashlib.md5(), hashlib.sha256()

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded original source read")
        data = self.source.read(size)
        self.count += len(data)
        self.md5.update(data)
        self.sha.update(data)
        if self.count > SIZE:
            raise ValueError("Oversized original MusicNet transfer")
        if self.count-self.progress >= 256*1024*1024:
            self.progress = self.count
            print(json.dumps({"compressed_bytes_read": self.count, "total": SIZE}), flush=True)
        return data

    def verify(self):
        if self.count != SIZE or self.md5.hexdigest() != MD5 or self.sha.hexdigest() != SHA256:
            raise ValueError("Incomplete or corrupt pinned original object")


def exact(source, size):
    if size < 0:
        raise ValueError("Negative source byte count")
    data = source.read(size)
    if len(data) != size:
        raise ValueError("Truncated original member")
    return data


def discard(source, size):
    if size < 0:
        raise ValueError("Overlapping or reversed source crop")
    while size:
        amount = min(size, 1024*1024)
        exact(source, amount)
        size -= amount


def safe_member(member):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or ".." in path.parts or "\\" in member.name
            or not (member.isdir() or member.isfile()) or not 0 <= member.size <= 2*1024**3):
        raise ValueError("Unsafe original archive member")


def crop_wave(source, size, starts):
    if (not starts or any(type(start) is not int or start < 0 for start in starts)
            or starts != sorted(set(starts))
            or any(b < a+SECONDS for a, b in zip(starts[:-1], starts[1:], strict=True))):
        raise ValueError("Invalid predetermined source crop clocks")
    header = exact(source, 12)
    if (header[:4] != b"RIFF" or header[8:] != b"WAVE"
            or struct.unpack("<I", header[4:8])[0]+8 != size):
        raise ValueError("Changed original RIFF container")
    position, fmt, result = 12, None, None
    while position < size:
        tag, length = struct.unpack("<4sI", exact(source, 8))
        position += 8
        if position+length+length%2 > size:
            raise ValueError("RIFF chunk escapes original member")
        if tag == b"fmt ":
            if fmt is not None or not 16 <= length <= 64:
                raise ValueError("Invalid original waveform format")
            fmt = exact(source, length)
            code, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if (code not in (1, 3) or channels not in (1, 2) or rate != RATE
                    or bits not in (16, 24, 32) or (code == 3 and bits != 32)
                    or align != channels*(bits//8) or byte_rate != rate*align):
                raise ValueError("Unsupported original sample clock")
        elif tag == b"data":
            if fmt is None or result is not None or length%align:
                raise ValueError("Invalid original PCM framing")
            duration = length/byte_rate
            if duration < starts[-1]+SECONDS:
                raise ValueError("Incomplete predetermined waveform crop")
            result, consumed = {}, 0
            for start in starts:
                first, amount = start*byte_rate, SECONDS*byte_rate
                discard(source, first-consumed)
                raw = exact(source, amount)
                chunks = b"fmt "+struct.pack("<I", len(fmt))+fmt
                chunks += b"data"+struct.pack("<I", len(raw))+raw
                result[start] = b"RIFF"+struct.pack("<I", len(chunks)+4)+b"WAVE"+chunks
                consumed = first+amount
            discard(source, length-consumed)
        else:
            discard(source, length)
        if length%2:
            exact(source, 1)
        position += length+length%2
    if result is None:
        raise ValueError("Missing original waveform samples")
    return result, {"sample_rate": rate, "channels": channels, "format_code": code,
                    "bits": bits, "original_duration": duration}


def verified_jobs(source, assessment):
    registered = json.loads(assessment.read_text())
    with tempfile.TemporaryDirectory(prefix="musicnet-later-audit-") as temporary:
        path = Path(temporary)/"audit.json"
        with contextlib.redirect_stdout(io.StringIO()):
            audit(source, path)
        if json.loads(path.read_text()) != registered:
            raise ValueError("Changed fixed later-passage symbolic assessment")
    jobs = registered["jobs"]
    originals = {row["id"]: row for row in json.loads((source/"manifest.json").read_text())["items"]}
    if (len(jobs) != 49 or len({job["id"] for job in jobs}) != 49
            or len({job["publisher_id"] for job in jobs}) != 20
            or any(job["group"] != "train" or originals[job["publisher_id"]]["group"] != "train" for job in jobs)):
        raise ValueError("Incomplete or leaking fixed later fitting population")
    return jobs, originals


def acquire(source, assessment, output):
    jobs, originals = verified_jobs(source, assessment)
    names = ("acquire_musicnet_later_v60.py", "audit_musicnet_expansion_v59.py")
    sources = {name: digest(Path(__file__).with_name(name)) for name in names}
    plan = {"version": VERSION, "source_root": str(source), "assessment_sha256": digest(assessment),
        "source_manifest_sha256": digest(source/"manifest.json"), "source_sha256": sources,
        "source_url": URL, "archive_bytes": SIZE, "archive_md5": MD5, "archive_sha256": SHA256,
        "jobs": jobs, "expected_recordings": 49, "expected_training_parents": 20,
        "historical_crops_must_be_bit_identical": True, "no_full_archive_stored": True,
        "no_model_inference": True, "no_user_audio_or_scores": True,
        "validation_reserved_and_consumed_members_not_retained_or_decoded": True,
        "no_change_to_v59_fitting_or_selection": True, "training_cleared": False}
    if output.exists():
        if (json.loads((output/"plan.json").read_text()) != plan or (output/"manifest.json").exists()):
            raise ValueError("Preserve completed or changed later-source acquisition")
    else:
        output.mkdir()
        (output/"audio").mkdir()
        (output/"labels").mkdir()
        preserve(output/"plan.json", plan)
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        publisher = json.loads(response.read(2*1024*1024))
    objects = [item for item in publisher["files"] if item["key"] == "musicnet.tar.gz"]
    if (publisher["id"] != 5120004 or publisher["metadata"]["license"]["id"] != "cc-by-4.0"
            or len(objects) != 1 or objects[0]["size"] != SIZE or objects[0]["checksum"] != "md5:"+MD5):
        raise ValueError("Changed publisher grant or original object identity")
    same_write(output/"publisher-grant.json", (json.dumps({"record": 5120004, "license": "CC BY 4.0",
        "key": "musicnet.tar.gz", "bytes": SIZE, "md5": MD5}, indent=2)+"\n").encode())
    by_parent = {}
    for job in jobs:
        by_parent.setdefault(job["publisher_id"], []).append(job)
    found_audio, found_labels, seen = {}, {}, set()
    with urllib.request.urlopen(URL, timeout=60) as remote:
        if remote.status != 200 or int(remote.headers["Content-Length"]) != SIZE:
            raise ValueError("Changed original archive transfer")
        checked = CheckedReader(remote)
        with gzip.GzipFile(fileobj=checked) as uncompressed:
            with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
                for member in archive:
                    safe_member(member)
                    if member.name in seen:
                        raise ValueError("Duplicate original member identity")
                    seen.add(member.name)
                    match = re.fullmatch(r"(?:\./)?musicnet/(train_data|train_labels)/(\d+)\.(wav|csv)", member.name)
                    if not match or match[2] not in by_parent:
                        continue
                    category, identity, extension = match.groups()
                    stream = archive.extractfile(member)
                    original, proposed = originals[identity], by_parent[identity]
                    if category == "train_data" and extension == "wav":
                        if any(job["id"] in found_audio for job in proposed):
                            raise ValueError("Duplicate original fitting waveform identity")
                        old = {row["start_sample"]//RATE: row for row in original["excerpts"]}
                        starts = sorted(set(old) | {job["start_sample"]//RATE for job in proposed})
                        waves, clock = crop_wave(stream, member.size, starts)
                        if clock != original["clock"] or member.name != original["original_member"]:
                            raise ValueError("Changed complete original waveform clock")
                        if any(hashlib.sha256(waves[start]).hexdigest() != row["audio_sha256"] for start, row in old.items()):
                            raise ValueError("Historical fitting crop bytes differ from verified original")
                        for job in proposed:
                            path = output/"audio"/(job["id"]+".wav")
                            same_write(path, waves[job["start_sample"]//RATE])
                            found_audio[job["id"]] = {**job, "audio": str(path), "audio_sha256": digest(path),
                                "clock": clock, "historical_source_crops_bit_identical": True}
                    elif category == "train_labels" and extension == "csv":
                        if identity in found_labels:
                            raise ValueError("Duplicate original fitting label identity")
                        if member.size > 8*1024*1024 or member.name != original["original_label_member"]:
                            raise ValueError("Changed original label member")
                        data = exact(stream, member.size)
                        if hashlib.sha256(data).hexdigest() != original["labels_sha256"]:
                            raise ValueError("Changed original fitting labels")
                        path = output/"labels"/(identity+".csv")
                        same_write(path, data)
                        found_labels[identity] = {"labels": str(path), "labels_sha256": digest(path)}
                    else:
                        raise ValueError("Changed original member semantics")
            while uncompressed.read(1024*1024):
                pass
        while checked.read(1024*1024):
            pass
        checked.verify()
    if (set(found_audio) != {job["id"] for job in jobs} or set(found_labels) != set(by_parent)
            or plan["source_manifest_sha256"] != digest(source/"manifest.json")
            or plan["assessment_sha256"] != digest(assessment)
            or sources != {name: digest(Path(__file__).with_name(name)) for name in names}):
        raise ValueError("Incomplete source acquisition or changed producer")
    records = [{**found_audio[job["id"]], **found_labels[job["publisher_id"]]} for job in jobs]
    preserve(output/"source.json", {"url": URL, "bytes": checked.count, "md5": checked.md5.hexdigest(),
        "sha256": checked.sha.hexdigest(), "complete_original_object_verified": True,
        "license": "CC BY 4.0", "publisher_record": RECORD, "no_full_archive_stored": True})
    preserve(output/"manifest.json", {"version": VERSION, "items": records,
        "plan_sha256": digest(output/"plan.json"), "source_sha256": digest(output/"source.json"),
        "no_model_inference": True, "no_user_audio_or_scores": True, "training_cleared": False,
        "all_parents_fitting_only": True, "historical_source_crops_bit_identical": True})
    print(json.dumps({"complete_original_object_verified": True, "recordings": len(records),
                      "training_parents": len(by_parent), "no_model_inference": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assessment", "output"):
        parser.add_argument(name, type=Path)
    acquire(**vars(parser.parse_args()))
