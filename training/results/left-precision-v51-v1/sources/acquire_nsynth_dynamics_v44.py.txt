"""Acquire unused NSynth velocities without opening reserved instrument audio."""

import argparse
import hashlib
import io
import json
import re
import tarfile
import wave
from collections import Counter
from pathlib import Path, PurePosixPath

from acquire_nsynth_subset import LICENSE, MD5, SIZE, URL, safe_member, select
from prepare_robust_training_stems import digest, preserve

VERSION = "nsynth-original-dynamics-v44-v1"
ARCHIVE_SHA256 = "00dea2645fbe0069258567da30807a90825e0bab54c077d6481f253096c4e2a0"
VELOCITIES = (25, 50, 100, 127)
EXPECTED = {"train": 1188, "validation": 327}


def verify_archive(archive):
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with archive.open("rb") as reader:
        for block in iter(lambda: reader.read(1024*1024), b""):
            md5.update(block)
            sha256.update(block)
            size += len(block)
    if size != SIZE or md5.hexdigest() != MD5 or sha256.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Incomplete or changed original NSynth bytes")
    return {"url": URL, "bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
            "license": "CC BY 4.0", "publisher_license_url": LICENSE, "publisher_split": "valid",
            "scope": "Project-specific existing instrument partitions; not an official NSynth benchmark."}


def selection(metadata):
    previous = select(metadata)
    groups = {n["instrument_str"]: n["group"] for n in previous.values()}
    chosen = {key: {**note, "group": groups[note["instrument_str"]]}
              for key, note in metadata.items()
              if note["instrument_str"] in groups
              and groups[note["instrument_str"]] in ("train", "validation")
              and 21 <= note["pitch"] < 60 and note["velocity"] in VELOCITIES}
    if dict(Counter(n["group"] for n in chosen.values())) != EXPECTED:
        raise ValueError("Changed unused-velocity coverage")
    return chosen


def acquire(archive, output):
    source = verify_archive(archive)
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("nsynth-valid/examples.json")
        safe_member(member)
        metadata = json.load(tar.extractfile(member))
    chosen = selection(metadata)
    output.mkdir(exist_ok=False)
    (output/"audio").mkdir()
    preserve(output/"source.json", source)
    preserve(output/"plan.json", {
        "version": VERSION, "source_sha256": digest(output/"source.json"),
        "producer_sha256": digest(Path(__file__)),
        "partition_producer_sha256": digest(Path(__file__).with_name("acquire_nsynth_subset.py")),
        "selection": "Original bass/keyboard MIDI 21-59, velocities 25/50/100/127. Same 11 train and 3 validation instruments as velocity-75 subset; reserved excluded.",
        "selected_ids": sorted(chosen), "expected_counts": EXPECTED,
        "no_reserved_audio_read": True, "no_decoder_inference": True,
        "no_user_audio_or_scores": True,
    })
    records, seen = [], set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            safe_member(member)
            if member.name in seen:
                raise ValueError("Duplicate NSynth member")
            seen.add(member.name)
            key = PurePosixPath(member.name).stem
            if not member.isfile() or member.name != f"nsynth-valid/audio/{key}.wav" or key not in chosen:
                continue
            if not re.fullmatch(r"[a-z0-9_]+-\d{3}-\d{3}", key) or member.size != 128044:
                raise ValueError("Changed NSynth PCM member")
            payload = tar.extractfile(member).read()
            with wave.open(io.BytesIO(payload)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 64000):
                    raise ValueError("Changed source sample clock")
            path = output/"audio"/(key+".wav")
            path.write_bytes(payload)
            records.append({"id": key, **chosen[key], "member": member.name,
                            "audio": str(path), "audio_sha256": digest(path), "bytes": len(payload)})
    if {n["id"] for n in records} != set(chosen):
        raise ValueError("Incomplete dynamics extraction")
    preserve(output/"manifest.json", {"version": VERSION, "items": sorted(records, key=lambda n: n["id"]),
        "plan_sha256": digest(output/"plan.json"), "no_reserved_audio_read": True,
        "no_decoder_inference": True, "no_user_audio_or_scores": True})
    (output/"ATTRIBUTION.txt").write_text("NSynth Dataset, Google Inc., CC BY 4.0.\n"
        "Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck, Karen Simonyan, Mohammad Norouzi.\n"
        "https://magenta.withgoogle.com/datasets/nsynth\nhttps://creativecommons.org/licenses/by/4.0/\n"
        "Subset of original samples at previously unused velocities; no audio modifications during acquisition.\n")
    print(json.dumps({"passes": True, "notes": len(records), "partitions": EXPECTED,
                      "manifest_sha256": digest(output/"manifest.json")}), flush=True)


def sources(folder):
    plan = json.loads((folder/"plan.json").read_text())
    source = json.loads((folder/"source.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    records = manifest["items"]
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or source["sha256"] != ARCHIVE_SHA256 or source["license"] != "CC BY 4.0"
            or plan["source_sha256"] != digest(folder/"source.json")
            or plan["producer_sha256"] != digest(Path(__file__))
            or plan["partition_producer_sha256"] != digest(Path(__file__).with_name("acquire_nsynth_subset.py"))
            or manifest["plan_sha256"] != digest(folder/"plan.json")
            or sorted(n["id"] for n in records) != plan["selected_ids"]
            or len(set(plan["selected_ids"])) != 1515
            or dict(Counter(n["group"] for n in records)) != EXPECTED
            or any(n["velocity"] not in VELOCITIES or not 21 <= n["pitch"] < 60 for n in records)
            or not all(plan[k] and manifest[k] for k in ("no_reserved_audio_read", "no_decoder_inference", "no_user_audio_or_scores"))):
        raise ValueError("Changed or leaking dynamics source")
    by_group = {g: {n["instrument_str"] for n in records if n["group"] == g} for g in EXPECTED}
    if len(by_group["train"]) != 11 or len(by_group["validation"]) != 3 or by_group["train"] & by_group["validation"]:
        raise ValueError("Instrument leakage in dynamics subset")
    for note in records:
        path = Path(note["audio"])
        if path.resolve() != (folder/"audio"/(note["id"]+".wav")).resolve() or digest(path) != note["audio_sha256"]:
            raise ValueError("Changed original dynamics waveform")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    acquire(**vars(parser.parse_args()))
