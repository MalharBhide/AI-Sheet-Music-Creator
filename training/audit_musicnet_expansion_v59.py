"""Audit fixed later training passages from licensed MusicNet metadata/labels only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

STARTS, SECONDS, RATE = (95, 155, 215), 20, 44100
SIZE = 11097394998
MD5 = "844764911fa0d5b97c97da944a057590"
SHA256 = "f26e6d023b7a62cd54d8f2b0533b88f549c30e3fad0aa546d1ea24aefba28fca"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposals(rows):
    jobs = []
    partitions = {group: {tuple(row["work"]) for row in rows if row["group"] == group}
                  for group in ("train", "validation", "reserved")}
    if (partitions["train"] & (partitions["validation"] | partitions["reserved"])
            or partitions["validation"] & partitions["reserved"]
            or len({row["id"] for row in rows}) != len(rows)
            or any(row["group"] not in partitions for row in rows)):
        raise ValueError("Changed or leaking original work partitions")
    for row in rows:
        if row["group"] != "train":
            continue
        duration = row["clock"]["original_duration"]
        if (row["ensemble"] != "Solo Piano" or row["clock"]["sample_rate"] != RATE
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("Invalid original solo-piano training clock")
        for start in STARTS:
            if start+SECONDS > duration:
                continue
            first, last = start*RATE, (start+SECONDS)*RATE
            if any(first < old["end_sample"] and last > old["start_sample"] for old in row["excerpts"]):
                raise ValueError("Later proposal overlaps an existing excerpt")
            jobs.append({"id": f"musicnet-later-v59-{row['id']}-{start}", "publisher_id": row["id"],
                "group": "train", "work": row["work"], "start_sample": first, "end_sample": last,
                "source_group": "musicnet/"+"/".join(row["work"]),
                "labels_sha256": row["labels_sha256"], "original_duration": duration,
                "original_member": row["original_member"]})
    return jobs


def label_summary(labels, job):
    first, last = job["start_sample"], job["end_sample"]
    complete, crossing, keys = 0, 0, set()
    for row in labels:
        start, end, pitch, instrument = [int(row[key]) for key in ("start_time", "end_time", "note", "instrument")]
        if (not 0 <= start < end <= round(job["original_duration"]*RATE)
                or not 21 <= pitch <= 108 or instrument != 1):
            raise ValueError("Invalid original solo-piano label clock or instrument")
        if start < last and end > first:
            keys.add(pitch)
            if first <= start and end <= last:
                complete += 1
            else:
                crossing += 1
    return {"complete_publisher_note_intervals": complete,
            "crossing_intervals_for_pitch_support_only": crossing, "distinct_pitches": len(keys)}


def audit(source, output):
    plan = json.loads((source/"plan.json").read_text())
    manifest = json.loads((source/"manifest.json").read_text())
    witness = json.loads((source/"source.json").read_text())
    publisher = json.loads((source/"publisher.json").read_text())
    objects = [record for record in publisher["files"] if record["key"] == "musicnet.tar.gz"]
    if (manifest["plan_sha256"] != digest(source/"plan.json")
            or manifest["source_sha256"] != digest(source/"source.json")
            or plan["producer_sha256"] != digest(Path(__file__).with_name("acquire_musicnet_subset.py"))
            or plan["publisher_sha256"] != digest(source/"publisher.json")
            or plan["metadata_sha256"] != digest(source/"metadata.csv")
            or witness["bytes"] != SIZE or witness["md5"] != MD5 or witness["sha256"] != SHA256
            or witness["complete_original_object_verified"] is not True
            or witness["license"] != "CC BY 4.0" or publisher["metadata"]["license"]["id"] != "cc-by-4.0"
            or len(objects) != 1 or objects[0]["size"] != SIZE or objects[0]["checksum"] != "md5:"+MD5
            or any(manifest[key] is not True for key in ("work_disjoint", "reserved_inference_not_performed", "no_user_audio_or_scores"))
            or plan["reserved_decoder_inference_not_allowed"] is not True):
        raise ValueError("Changed licensed original MusicNet provenance")
    rows = manifest["items"]
    selected = {row["id"]: row for row in plan["selected"]}
    if (len(rows) != 32 or len(selected) != 32
            or any(any(row[key] != value for key, value in selected[row["id"]].items()) for row in rows)
            or sum(row["group"] == "train" for row in rows) != 21):
        raise ValueError("Changed fixed original training work selection")
    jobs = proposals(rows)
    notes = {}
    for row in rows:
        if row["group"] != "train":
            continue
        path = source/"labels"/(row["id"]+".csv")
        if (digest(path) != row["labels_sha256"]
                or row["original_member"] != "musicnet/train_data/"+row["id"]+".wav"):
            raise ValueError("Changed original fitting labels or waveform member")
        with path.open() as stream:
            notes[row["id"]] = list(csv.DictReader(stream))
    jobs = [{**job, **label_summary(notes[job["publisher_id"]], job)} for job in jobs]
    result = {"version": "musicnet-later-fitting-symbolic-audit-v59-v1", "source_manifest_sha256": digest(source/"manifest.json"),
        "audit_producer_sha256": digest(Path(__file__)), "source_archive_sha256": SHA256,
        "source_archive_bytes": SIZE, "license": "CC BY 4.0", "training_parent_works": 21,
        "proposed_recordings": len(jobs), "fixed_start_seconds": list(STARTS), "duration_seconds": SECONDS,
        "publisher_note_intervals": sum(job["complete_publisher_note_intervals"] for job in jobs),
        "crossing_intervals_for_pitch_support_only": sum(job["crossing_intervals_for_pitch_support_only"] for job in jobs),
        "no_source_audio_read_or_acquired": True, "no_model_inference": True, "no_user_audio_or_scores": True,
        "no_validation_reserved_or_consumed_labels_read": True, "training_cleared": False,
        "new_original_waveform_acquisition_and_clock_preflight_required": True,
        "no_change_to_v58_fitting_or_selection": True, "publisher_labels_not_sustain_pedal_or_acoustic_release_truth": True,
        "jobs": jobs}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "jobs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    audit(**vars(parser.parse_args()))
