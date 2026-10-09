"""Read only fully verified sources; use synthesis-control clocks, never predictions."""

import csv
import json
import math
from pathlib import Path

from acquire_cocochorales_subset import ASSESSMENT_SHA, select
from assess_cocochorales import checksum_list
from prepare_robust_training_stems import digest

FRAME_RATE = 250


def sources(root):
    plan = json.loads((root / "plan.json").read_text())
    assessment_root = Path(plan["assessment_root"])
    if (
        plan["assessment_sha256"] != ASSESSMENT_SHA
        or digest(assessment_root / "assessment.json") != ASSESSMENT_SHA
        or plan["producer_sha256"]
        != digest(Path(__file__).with_name("acquire_cocochorales_subset.py"))
        or digest(root / "publisher.html") != plan["publisher_sha256"]
        or plan["license"] != "CC BY 4.0"
        or not plan["no_fitting_inference_or_user_audio"]
    ):
        raise ValueError("Changed or unverified CocoChorales acquisition")
    assessment = json.loads((assessment_root / "assessment.json").read_text())
    sums_path = assessment_root / "publisher-md5.txt"
    if (
        digest(sums_path) != plan["publisher_md5_list_sha256"]
        or select(assessment) != plan["selected"]
    ):
        raise ValueError("Changed publisher checksum or source partition")
    sums = checksum_list(sums_path.read_bytes())
    witnesses, result = {}, []
    for group in ("train", "validation"):
        publisher_group = "train" if group == "train" else "valid"
        for component in ("note_expression", "main_dataset"):
            witness_path = root / f"{group}-{component}-verified.json"
            if not witness_path.exists():
                raise ValueError("Complete original checksums are required before observation")
            witness = json.loads(witness_path.read_text())
            expected = next(
                o
                for o in plan["objects"]
                if o["name"] == f"{component}/{publisher_group}/1.tar.bz2"
            )
            if (
                witness["object"] != expected
                or witness["plan_sha256"] != digest(root / "plan.json")
                or witness["bytes"] != expected["bytes"]
                or expected["md5"] != sums[expected["name"]]
                or not witness["complete_md5_and_decompression_footer_verified"]
                or not witness["no_fitting_inference_or_user_audio"]
            ):
                raise ValueError("Changed or incomplete publisher object witness")
            for name, row in witness["retained"].items():
                path = root / "retained" / group / component / name
                if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                    raise ValueError("Changed retained publisher bytes")
            witnesses[(group, component)] = witness
    for row in plan["selected"]:
        folder = root / "retained" / row["group"]
        files = {}
        for component in ("main_dataset", "note_expression"):
            names = {
                name: data
                for name, data in witnesses[(row["group"], component)]["retained"].items()
                if name.split("/")[0] == row["id"]
            }
            if len(names) != (11 if component == "main_dataset" else 4):
                raise ValueError("Incomplete original source and four-part labels")
            files[component] = {
                name: {**data, "path": str(folder / component / name)}
                for name, data in names.items()
            }
        result.append({**row, "files": files})
    if len(result) != 18 or len({r["original_midi"] for r in result}) != 18:
        raise ValueError("Incomplete or leaking source partition")
    return result


def note_rows(path):
    result, previous_end = [], 0
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        if not {"pitch", "onset", "offset", "note_length"}.issubset(reader.fieldnames or ()):
            raise ValueError("Missing synthesis-control timestamp columns")
        for row in reader:
            values = [float(row[k]) for k in ("pitch", "onset", "offset", "note_length")]
            if any(not math.isfinite(v) or v != int(v) for v in values):
                raise ValueError("Noninteger or nonfinite synthesis clock")
            pitch, start, end, length = map(int, values)
            if (
                not 0 <= pitch <= 127
                or start < previous_end
                or end < start
                or length != end - start
                or (pitch != 0 and length <= 0)
            ):
                raise ValueError("Invalid monophonic synthesis interval")
            previous_end = end
            if pitch:
                if not 21 <= pitch <= 108:
                    raise ValueError("Unsupported labeled instrument register")
                result.append([start / FRAME_RATE, end / FRAME_RATE, pitch])
    if not result:
        raise ValueError("Empty labeled source part")
    return result


def references(rows, duration, original_duration):
    if not math.isfinite(duration) or not 5 <= duration <= original_duration:
        raise ValueError("Invalid excerpt or original waveform clock")
    attacks, support = [], []
    for start, end, pitch in rows:
        if not 0 <= start < end <= original_duration + 1 / FRAME_RATE:
            raise ValueError("Synthesis labels escape original waveform clock")
        if start >= duration:
            continue
        attacks.append([start, min(end, duration), pitch])
        # Preserve synthesis/reverb decay without claiming an exact key release.
        support.append([max(0, start - 0.05), min(duration, end + 1.0), pitch])
    return attacks, support
