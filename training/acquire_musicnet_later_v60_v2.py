"""Separate bounded parallel acquisition of the same fixed original MusicNet object."""

import argparse
import gzip
import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

from acquire_musicnet_later_v60 import (
    MD5,
    RECORD,
    SHA256,
    SIZE,
    URL,
    CheckedReader,
    crop_wave,
    digest,
    exact,
    preserve,
    safe_member,
    same_write,
    verified_jobs,
)
from audit_musicnet_expansion_v59 import RATE
from parallel_archive_v60 import ParallelArchive

VERSION = "musicnet-later-original-fitting-source-v60-v2"


def acquire(source, assessment, output):
    jobs, originals = verified_jobs(source, assessment)
    names = ("acquire_musicnet_later_v60_v2.py", "acquire_musicnet_later_v60.py", "parallel_archive_v60.py", "audit_musicnet_expansion_v59.py")
    sources = {name: digest(Path(__file__).with_name(name)) for name in names}
    plan = {"version": VERSION, "source_root": str(source), "assessment_sha256": digest(assessment),
        "source_manifest_sha256": digest(source/"manifest.json"), "source_sha256": sources,
        "source_url": URL, "archive_bytes": SIZE, "archive_md5": MD5, "archive_sha256": SHA256,
        "jobs": jobs, "expected_recordings": 49, "expected_training_parents": 20,
        "historical_crops_must_be_bit_identical": True, "no_full_archive_stored": True,
        "transport": {"workers": 4, "chunk_bytes": 16*1024*1024, "maximum_read_bytes": 1024*1024},
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
    with ParallelArchive(URL, SIZE, workers=4, chunk_bytes=16*1024*1024) as remote:
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
