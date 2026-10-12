"""Acquire unambiguous, progression-disjoint licensed NRG-CP MIDI sources."""

import argparse
import collections
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import mido

VERSION = "nrgcp-unambiguous-progression-source-v47-v1"
PUBLISHER = "https://zenodo.org/records/15304989"
ARCHIVE_MD5 = "7443fe30674ef149aa4c23580044f597"
ARCHIVE_SHA256 = "ceb0f0b8d808150364a1ded15a1f54f9dc3a66f6f51dc0708c8242b62ae2f382"
EXPECTED_FILES, EXPECTED_AMBIGUOUS = 30943, 1426
COUNTS = {"train": 48, "validation": 12}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as writer:
        json.dump(value, writer, indent=2)


def member_path(member):
    path = PurePosixPath(member.name)
    if (not member.isfile() or path.is_absolute() or len(path.parts) != 1
            or ".." in path.parts or "\\" in member.name or not member.name.endswith("_nrgcp_dataset.mid")
            or not 14 < member.size <= 16384):
        raise ValueError("Unsafe or unexpected NRG-CP member")
    return member.name


def notes(data):
    """Reject ambiguous same-key pairings instead of manufacturing durations."""
    midi = mido.MidiFile(file=io.BytesIO(data))
    if midi.type != 1 or midi.ticks_per_beat != 96:
        raise ValueError("Changed NRG-CP clock header")
    active, result, tick = {}, [], 0
    tempo, meter, programs = [], [], []
    for message in mido.merge_tracks(midi.tracks):
        if not isinstance(message.time, int) or message.time < 0:
            raise ValueError("Invalid MIDI delta clock")
        tick += message.time
        if message.type == "set_tempo":
            tempo.append((tick, message.tempo))
        elif message.type == "time_signature":
            meter.append((tick, message.numerator, message.denominator))
        elif message.type == "program_change":
            programs.append(message.program)
        elif message.type == "note_on":
            key = message.channel, message.note
            if message.channel == 9:
                raise ValueError("Drum channel is not a piano-key reference")
            if message.velocity:
                if key in active:
                    return None  # Entire clip excluded; no guessed FIFO/LIFO repair.
                active[key] = (tick, message.velocity)
            else:
                if key not in active:
                    raise ValueError("Unmatched NRG-CP note-off")
                start, velocity = active.pop(key)
                if tick <= start:
                    raise ValueError("Nonpositive NRG-CP note")
                result.append((start, tick, message.note, velocity))
        elif message.type != "end_of_track":
            raise ValueError("Unexpected MIDI message or controller")
    if (active or not result or tempo != [(0, 500000)] or meter != [(0, 4, 4)]
            or len(programs) != 1 or not 2689 <= tick <= 3073):
        raise ValueError("Incomplete or changed NRG-CP symbolic reference")
    return sorted(result)


def family(events):
    bars = [set() for _ in range(8)]
    for start, _, pitch, _ in events:
        bars[min(7, start//384)].add(pitch % 12)
    canonical = min(tuple(tuple(sorted((p-shift) % 12 for p in bar)) for bar in bars)
                    for shift in range(12))
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def pattern(events):
    root = min(e[2] for e in events)
    normalized = sorted((a, b, p-root) for a, b, p, _ in events)
    return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()


def partition(identifier):
    bucket = int(identifier[:16], 16) % 10
    return "reserved" if bucket == 0 else "validation" if bucket < 3 else "train"


def acquire(archive, output):
    raw = archive.read_bytes()
    if hashlib.md5(raw).hexdigest() != ARCHIVE_MD5 or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Changed publisher archive")
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "publisher": PUBLISHER, "license": "CC BY 4.0",
        "archive_md5": ARCHIVE_MD5, "archive_sha256": ARCHIVE_SHA256,
        "producer_sha256": digest(Path(__file__)), "counts": COUNTS,
        "selection": "Reject entire same-key-overlap clips. Group by transposition-invariant eight-bar pitch-class progression. Partition hash buckets 0 reserved, 1-2 validation, 3-9 train. Select smallest 48/12 eligible family hashes; within a family choose smallest source byte hash. No reserved extraction, rendering or decoder inference. Known exact transposition patterns must also be disjoint. Coarse musical grouping is documented, not a claim of human work IDs.",
        "no_audio_rendered_or_inferred": True, "no_user_audio_or_scores": True}
    save(output/"plan.json", plan)
    groups, names = collections.defaultdict(list), set()
    ambiguous = 0
    with tarfile.open(archive, "r|gz") as reader:
        for member in reader:
            name = member_path(member)
            if name in names:
                raise ValueError("Duplicate source path")
            names.add(name)
            data = reader.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError("Truncated NRG-CP MIDI")
            events = notes(data)
            if events is None:
                ambiguous += 1
                continue
            identifier = family(events)
            groups[identifier].append({"member": name, "sha256": hashlib.sha256(data).hexdigest(),
                "pattern": pattern(events), "notes": len(events)})
    if len(names) != EXPECTED_FILES or ambiguous != EXPECTED_AMBIGUOUS:
        raise ValueError("Changed source coverage or ambiguity count")
    selected = []
    for group, count in COUNTS.items():
        families = sorted(k for k in groups if partition(k) == group)[:count]
        if len(families) != count:
            raise ValueError("Insufficient independent progression families")
        selected.extend({**min(groups[k], key=lambda r: r["sha256"]), "family": k, "group": group}
                        for k in families)
    if ({r["pattern"] for r in selected if r["group"] == "train"}
            & {r["pattern"] for r in selected if r["group"] == "validation"}):
        raise ValueError("Cross-partition transposed note pattern")
    desired = {r["member"]: r for r in selected}
    records = []
    (output/"midi").mkdir()
    with tarfile.open(archive, "r|gz") as reader:
        for member in reader:
            if member.name not in desired:
                continue
            data = reader.extractfile(member).read()
            row = desired.pop(member.name)
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Changed selected source")
            identifier = "nrgcp-v47-"+row["sha256"][:24]
            path = output/"midi"/(identifier+".mid")
            path.write_bytes(data)
            records.append({**row, "id": identifier, "midi": str(path),
                "source_group": "nrgcp-v1/harmonic-"+row["family"]})
    if desired or len(records) != sum(COUNTS.values()) or digest(Path(__file__)) != plan["producer_sha256"]:
        raise ValueError("Incomplete or changed acquisition")
    save(output/"manifest.json", {"version": VERSION, "items": sorted(records, key=lambda r: r["id"]),
        "plan_sha256": digest(output/"plan.json"), "ambiguous_clips_excluded": ambiguous,
        "valid_progression_families": len(groups), "no_reserved_midi_extracted": True,
        "no_audio_rendered_or_inferred": True, "no_user_audio_or_scores": True})
    (output/"ATTRIBUTION.txt").write_text("NRG-CP v1, WaivOps / Patchbanks, 2025, CC BY 4.0.\n"
        "https://zenodo.org/records/15304989\nhttps://creativecommons.org/licenses/by/4.0/\n"
        "Unmodified selected MIDI bytes; ambiguous source clips excluded.\n")
    print(json.dumps({"acquired": len(records), "counts": COUNTS, "ambiguous_excluded": ambiguous}), flush=True)


def sources(folder):
    plan, manifest = [json.loads((folder/n).read_text()) for n in ("plan.json", "manifest.json")]
    rows = manifest["items"]
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["publisher"] != PUBLISHER or plan["license"] != "CC BY 4.0"
            or plan["archive_md5"] != ARCHIVE_MD5 or plan["archive_sha256"] != ARCHIVE_SHA256
            or plan["producer_sha256"] != digest(Path(__file__)) or plan["counts"] != COUNTS
            or manifest["plan_sha256"] != digest(folder/"plan.json")
            or manifest["ambiguous_clips_excluded"] != EXPECTED_AMBIGUOUS
            or len(rows) != sum(COUNTS.values()) or len({r["id"] for r in rows}) != len(rows)
            or len({r["family"] for r in rows}) != len(rows)
            or dict(collections.Counter(r["group"] for r in rows)) != COUNTS
            or not manifest["no_reserved_midi_extracted"]
            or not all(d[k] for d in (plan, manifest) for k in ("no_audio_rendered_or_inferred", "no_user_audio_or_scores"))):
        raise ValueError("Changed or leaking NRG-CP source contract")
    for row in rows:
        path = folder/"midi"/(row["id"]+".mid")
        if path.resolve() != Path(row["midi"]).resolve() or digest(path) != row["sha256"]:
            raise ValueError("Changed NRG-CP source MIDI")
        events = notes(path.read_bytes())
        if (events is None or len(events) != row["notes"] or family(events) != row["family"]
                or pattern(events) != row["pattern"] or partition(row["family"]) != row["group"]
                or row["source_group"] != "nrgcp-v1/harmonic-"+row["family"]):
            raise ValueError("Changed source note clock or progression group")
    if ({r["pattern"] for r in rows if r["group"] == "train"}
            & {r["pattern"] for r in rows if r["group"] == "validation"}):
        raise ValueError("Leaking transposed pattern")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    acquire(**vars(parser.parse_args()))
