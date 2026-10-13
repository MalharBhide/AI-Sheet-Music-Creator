"""Acquire the licensed TinySOL v6 pitch-presence source, without timing labels."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
import wave
from collections import Counter
from pathlib import Path, PurePosixPath

VERSION = "tinysol-pitch-source-v45-v1"
PUBLISHER = "https://zenodo.org/records/3685367"
ARCHIVE_MD5 = "36030a7fe389da86c3419e5ee48e3b7f"
METADATA_MD5 = "a86c9bb115f69e61f2f25872e397fc4a"
PARTITIONS = {
    "train": ("Bass Tuba", "Trombone", "Accordion", "Contrabass", "Viola"),
    "validation": ("Cello", "Bassoon"),
}
EXPECTED = {"train": 712, "validation": 237}


def checksums(path):
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def safe_path(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or "\x00" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe TinySOL path")
    return path.as_posix()


def safe_member(member):
    name = safe_path(member.name)
    if not (member.isfile() or member.isdir()) or not 0 <= member.size <= 16 * 1024 * 1024:
        raise ValueError("Unsafe TinySOL member type or size")
    return name


def selected_rows(rows):
    if len(rows) != 2913 or len({safe_path(row["Path"]) for row in rows}) != len(rows):
        raise ValueError("Changed TinySOL metadata coverage or duplicate paths")
    groups = {instrument: group for group, instruments in PARTITIONS.items() for instrument in instruments}
    if sum(map(len, PARTITIONS.values())) != len(groups):
        raise ValueError("Overlapping whole-instrument partitions")
    selected = {}
    for row in rows:
        path = safe_path(row["Path"])
        pitch = int(row["Pitch ID"])
        if not 0 <= pitch <= 127 or row["Needed digital retuning"] not in ("TRUE", "FALSE"):
            raise ValueError("Invalid supplied pitch or retuning metadata")
        instrument = row["Instrument (in full)"]
        if instrument not in groups or not 21 <= pitch < 60:
            continue
        if row["Technique (abbr.)"] != "ord" or row["Dynamics"] not in ("pp", "mf", "ff"):
            raise ValueError("Changed selected TinySOL technique or dynamics")
        selected[path] = {"id": "tinysol-v45-" + hashlib.sha256(path.encode()).hexdigest()[:24],
            "group": groups[instrument], "source_group": "tinysol-v6/" + instrument,
            "pitch": pitch, "dynamics": row["Dynamics"],
            "digitally_retuned": row["Needed digital retuning"] == "TRUE", "metadata": row}
    if dict(Counter(row["group"] for row in selected.values())) != EXPECTED:
        raise ValueError("Changed TinySOL fitting coverage")
    return selected


def pcm_clock(payload):
    with wave.open(io.BytesIO(payload)) as reader:
        clock = {"channels": reader.getnchannels(), "sample_width": reader.getsampwidth(),
                 "sample_rate": reader.getframerate(), "frames": reader.getnframes()}
        if (clock["channels"], clock["sample_width"], clock["sample_rate"]) != (1, 2, 44100):
            raise ValueError("Changed TinySOL PCM format")
        if not 0 < clock["frames"] <= 30 * clock["sample_rate"]:
            raise ValueError("Empty or unexpectedly long TinySOL recording")
        if len(reader.readframes(clock["frames"])) != clock["frames"] * 2:
            raise ValueError("Truncated TinySOL PCM data")
    return {**clock, "duration": clock["frames"] / clock["sample_rate"]}


def save(path, content):
    with path.open("x") as writer:
        writer.write(json.dumps(content, indent=2) + "\n")


def acquire(archive, metadata, output):
    source, supplied = checksums(archive), checksums(metadata)
    if source["md5"] != ARCHIVE_MD5 or supplied["md5"] != METADATA_MD5:
        raise ValueError("Publisher TinySOL checksum mismatch")
    with metadata.open(newline="") as reader:
        selected = selected_rows(list(csv.DictReader(reader)))
    output.mkdir(exist_ok=False)
    (output / "audio").mkdir()
    (output / "metadata.csv").write_bytes(metadata.read_bytes())
    save(output / "plan.json", {"version": VERSION, "publisher": PUBLISHER,
        "license": "CC BY 4.0", "source": source, "metadata": supplied,
        "producer_sha256": checksums(Path(__file__))["sha256"], "partitions": PARTITIONS,
        "selected_paths": sorted(selected), "expected_counts": EXPECTED,
        "supervision_scope": "Pitch presence only; precise human onset, key release and pedal times unavailable. Protect complete source clip at supplied fundamental. No invented timing labels or timing evaluation.",
        "no_reserved_audio_extracted_or_decoded": True, "no_audio_inference": True,
        "no_user_audio_or_scores": True})
    records, seen, found = [], set(), set()
    with tarfile.open(archive, "r:gz") as reader:
        for member in reader:
            name = safe_member(member)
            if name in seen:
                raise ValueError("Duplicate TinySOL archive path")
            seen.add(name)
            relative = name.removeprefix("TinySOL/")
            if not member.isfile() or relative not in selected:
                continue
            if relative in found:
                raise ValueError("Duplicate selected TinySOL recording")
            found.add(relative)
            payload = reader.extractfile(member).read()
            if len(payload) != member.size:
                raise ValueError("Truncated TinySOL member")
            clock = pcm_clock(payload)
            record = selected[relative]
            path = output / "audio" / (record["id"] + ".wav")
            with path.open("xb") as writer:
                writer.write(payload)
            records.append({**record, **clock, "member": name, "audio": str(path),
                "audio_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    if found != set(selected):
        raise ValueError("Incomplete TinySOL acquisition")
    save(output / "manifest.json", {"version": VERSION, "items": sorted(records, key=lambda r: r["id"]),
        "plan_sha256": checksums(output / "plan.json")["sha256"],
        "no_reserved_audio_extracted_or_decoded": True, "no_audio_inference": True,
        "no_user_audio_or_scores": True})
    (output / "ATTRIBUTION.txt").write_text("TinySOL v6, CC BY 4.0.\n"
        "Carmine-Emanuele Cella, Daniele Ghisi, Vincent Lostanlen, Fabien Lévy, Joshua Fineberg, Yan Maresz.\n"
        "https://zenodo.org/records/3685367\nhttps://creativecommons.org/licenses/by/4.0/\n"
        "Subset of supplied original bytes; publisher retuning flags retained. Recordings originate from Ircam SOL.\n")
    print(json.dumps({"passes": True, "recordings": len(records), "partitions": EXPECTED,
        "manifest_sha256": checksums(output / "manifest.json")["sha256"]}), flush=True)


def sources(folder):
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    metadata = checksums(folder / "metadata.csv")
    with (folder / "metadata.csv").open(newline="") as reader:
        selected = selected_rows(list(csv.DictReader(reader)))
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["license"] != "CC BY 4.0" or plan["publisher"] != PUBLISHER
            or plan["source"]["md5"] != ARCHIVE_MD5 or metadata["md5"] != METADATA_MD5
            or plan["metadata"] != metadata
            or plan["producer_sha256"] != checksums(Path(__file__))["sha256"]
            or plan["partitions"] != {k: list(v) for k, v in PARTITIONS.items()}
            or manifest["plan_sha256"] != checksums(folder / "plan.json")["sha256"]
            or plan["selected_paths"] != sorted(selected) or plan["expected_counts"] != EXPECTED
            or not all(doc[k] for doc in (plan, manifest) for k in
                ("no_reserved_audio_extracted_or_decoded", "no_audio_inference", "no_user_audio_or_scores"))):
        raise ValueError("Changed TinySOL acquisition contract")
    records = manifest["items"]
    if (len(records) != len(selected) or len({r["id"] for r in records}) != len(records)
            or dict(Counter(r["group"] for r in records)) != EXPECTED
            or {r["member"].removeprefix("TinySOL/") for r in records} != set(selected)):
        raise ValueError("Incomplete or leaking TinySOL source")
    for record in records:
        expected = selected[record["member"].removeprefix("TinySOL/")]
        path = Path(record["audio"])
        if (any(record[k] != v for k, v in expected.items())
                or path.resolve() != (folder / "audio" / (record["id"] + ".wav")).resolve()
                or checksums(path)["sha256"] != record["audio_sha256"]):
            raise ValueError("Changed TinySOL waveform or supplied metadata")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    acquire(**vars(parser.parse_args()))
