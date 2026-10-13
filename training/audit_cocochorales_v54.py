"""Audit training-only symbolic shards; no audio, pickle, extraction or fitting."""

import argparse
import bz2
import collections
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath

import yaml

FILES = {"expression": (10623029, "ed8b8a526ef39e5ad7a8b8e10efaa7bc"),
         "metadata": (2361634, "0345dda2f41cefefb13dc3eb7e9fa7fa")}
EXPRESSIONS = ("volume", "vol_fluc", "vibrato", "brightness", "attack", "vol_peak_pos")
VOICES = {0: "violin", 1: "violin", 2: "viola", 3: "cello"}


def members(path, kind):
    seen = set()
    # Publisher uses pbzip2: concatenated bzip streams need BZ2File, rather
    # than tarfile's single-stream r|bz2 decompressor.
    with bz2.BZ2File(path) as decoded, tarfile.open(fileobj=decoded, mode="r|") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or str(name) in seen):
                raise ValueError("Unsafe or duplicate symbolic member")
            seen.add(str(name))
            if member.isdir() and (str(name) == "." or re.fullmatch(r"string_track\d{6}", str(name))):
                continue
            pattern = (r"string_track(\d{6})/([0-3])_(violin|viola|cello)\.csv"
                if kind == "expression" else r"string_track(\d{6})\.yaml")
            match = re.fullmatch(pattern, str(name))
            if (not member.isfile() or not match or not 0 < member.size <= 1024 * 1024
                    or not 1 <= int(match[1]) <= 2000):
                raise ValueError("Unexpected source type, partition, ID or size")
            if kind == "expression" and VOICES[int(match[2])] != match[3]:
                raise ValueError("Changed voice/instrument mapping")
            raw = archive.extractfile(member).read()
            if len(raw) != member.size:
                raise ValueError("Truncated symbolic source")
            yield int(match[1]), int(match[2]) if kind == "expression" else None, raw


def voice_events(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if not rows or set(rows[0]) != {"", "pitch", "onset", "offset", "note_length", *EXPRESSIONS}:
        raise ValueError("Changed expression schema")
    result, previous = [], 0
    for index, row in enumerate(rows):
        values = {key: float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite expression annotation")
        integers = ("", "pitch", "onset", "offset", "note_length")
        if any(values[key] != int(values[key]) for key in integers):
            raise ValueError("Nonintegral source frame clock or pitch")
        pitch, start, end, length = [int(values[key]) for key in ("pitch", "onset", "offset", "note_length")]
        sentinel = index == 0 and pitch == start == end == length == 0
        if (values[""] != index or start < previous or length != end - start
                or (length <= 0 and not sentinel) or end > 30000
                or any(not 0 <= values[key] <= 1 for key in EXPRESSIONS)
                or (pitch != 0 and not 21 <= pitch <= 108)
                or (pitch == 0 and any(values[key] != 0 for key in EXPRESSIONS))):
            raise ValueError("Ambiguous or unsupported expression clock")
        previous = end
        if pitch:
            result.append((start, end, pitch))
    if not result:
        raise ValueError("Empty voiced source")
    return result


def piano_unambiguous(events):
    by_pitch = collections.defaultdict(list)
    for start, end, pitch in events:
        by_pitch[pitch].append((start, end))
    for values in by_pitch.values():
        values.sort()
        if any(a[1] > b[0] for a, b in zip(values, values[1:], strict=False)):
            return False
    return True


def audit(root):
    proof, voices, metadata = {}, collections.defaultdict(dict), {}
    for kind in FILES:
        path = root / f"{kind}-train-1.tar.bz2"
        raw = path.read_bytes()
        size, md5 = FILES[kind]
        if len(raw) != size or hashlib.md5(raw).hexdigest() != md5:
            raise ValueError("Publisher symbolic checksum or size mismatch")
        proof[kind] = {"bytes": size, "md5": md5, "sha256": hashlib.sha256(raw).hexdigest()}
        for identifier, voice, data in members(path, kind):
            if kind == "expression":
                voices[identifier][voice] = data
            else:
                value = yaml.safe_load(data)
                if (not isinstance(value, dict) or value.get("ensemble") != "string"
                        or value.get("instrument_name") != VOICES
                        or not re.fullmatch(r"\d+\.mid", str(value.get("midi_file")))):
                    raise ValueError("Changed source grouping or ensemble metadata")
                metadata[identifier] = value
    if set(voices) != set(metadata) or set(voices) != set(range(1, 2001)) or any(set(v) != set(VOICES) for v in voices.values()):
        raise ValueError("Incomplete four-voice training shard")
    counts = collections.Counter()
    exclusions = collections.Counter()
    groups = set()
    for identifier, sources in voices.items():
        counts["chorales"] += 1
        groups.add(metadata[identifier]["midi_file"])
        try:
            events = [event for data in sources.values() for event in voice_events(data)]
        except ValueError as error:
            exclusions[str(error)] += 1
            continue
        counts["valid_four_voice_clocks"] += 1
        counts["source_notes"] += len(events)
        counts["holds_over_1s"] += sum(end - start > 250 for start, end, _ in events)
        counts["holds_over_4s"] += sum(end - start > 1000 for start, end, _ in events)
        unambiguous = piano_unambiguous(events)
        counts["unambiguous_piano_key_clocks"] += int(unambiguous)
        counts["same_key_ambiguous_chorales"] += int(not unambiguous)
    return {"version": "cocochorales-training-symbolic-audit-v54-v1",
        "publisher": "https://magenta.tensorflow.org/datasets/cocochorales", "license": "CC BY 4.0",
        "inputs": proof, "counts": dict(counts), "excluded_clock_reasons": dict(exclusions),
        "distinct_parent_midi_ids": len(groups), "expression_clock_seconds_per_frame": .004,
        "audio_or_pickle_acquired": False, "audio_rendered_or_inferred": False,
        "validation_or_test_shards_acquired": False, "user_audio_or_scores_processed": False,
        "training_cleared": False, "new_weights_deployed": False,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Source assessment only; no selection, rendering, fitting or release evidence.",
            "Generated string ensembles are not human piano timing or actual piano recordings.",
            "Original audio has per-note pitch corrections; nominal MIDI alone is not exact audio pitch truth.",
            "Any future authored-piano derivative needs separately frozen labels, key clocks and composer/source grouping.",
            "Reject ambiguous same-key clocks; do not guess voice merges or release durations."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open("x") as writer:
        json.dump(result, writer, indent=2)
    print(json.dumps(result), flush=True)
