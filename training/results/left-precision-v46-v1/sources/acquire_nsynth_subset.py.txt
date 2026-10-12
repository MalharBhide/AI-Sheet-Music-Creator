"""Verify original NSynth archive and retain a bounded instrument-disjoint subset."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

from prepare_robust_training_stems import digest, preserve

URL = "https://storage.googleapis.com/download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
MD5 = "87e94a00a19b6dbc99cf6d4c0c0cae87"
SIZE = 1068767009
LICENSE = "https://magenta.withgoogle.com/datasets/nsynth"


def verified(archive):
    md5 = hashlib.md5()
    size = 0
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            md5.update(block)
    if size != SIZE or md5.hexdigest() != MD5:
        raise ValueError("Incomplete or changed original NSynth archive")
    return {
        "url": URL,
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": digest(archive),
        "license": "CC BY 4.0",
        "publisher_license_url": LICENSE,
        "publisher_split": "valid",
        "scope": "Project-specific instrument split of the original validation archive. Not an official NSynth benchmark.",
    }


def select(metadata):
    if len(metadata) != 12678:
        raise ValueError("Changed NSynth original validation metadata coverage")
    selected = {
        key: value
        for key, value in metadata.items()
        if value["instrument_family_str"] in ("bass", "keyboard")
        and 21 <= value["pitch"] < 60
        and value["velocity"] == 75
    }
    instruments = {
        family: sorted(
            {v["instrument_str"] for v in selected.values() if v["instrument_family_str"] == family}
        )
        for family in ("bass", "keyboard")
    }
    if {k: len(v) for k, v in instruments.items()} != {"bass": 10, "keyboard": 8}:
        raise ValueError("Unexpected NSynth source instrument coverage")
    groups = {}
    for _family, ids in instruments.items():
        train = int(0.65 * len(ids))
        validation = max(1, int(0.2 * len(ids)))
        for index, name in enumerate(ids):
            groups[name] = (
                "train"
                if index < train
                else "validation"
                if index < train + validation
                else "reserved"
            )
    return {
        key: {**value, "group": groups[value["instrument_str"]]} for key, value in selected.items()
    }


def safe_member(member):
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or member.issym()
        or member.islnk()
        or not (member.isfile() or member.isdir())
        or member.size > 16 * 1024 * 1024
    ):
        raise ValueError("Unsafe or unexpectedly large NSynth archive member")


def acquire(archive, output):
    source = verified(archive)
    with tarfile.open(archive, "r:gz") as tar:
        metadata = None
        for member in tar:
            safe_member(member)
            if member.name == "nsynth-valid/examples.json":
                metadata = json.load(tar.extractfile(member))
                break
    if metadata is None:
        raise ValueError("Missing original NSynth metadata")
    selected = select(metadata)
    output.mkdir(exist_ok=False)
    folder = output / "audio"
    folder.mkdir()
    preserve(output / "source.json", source)
    preserve(
        output / "plan.json",
        {
            "source_sha256": digest(output / "source.json"),
            "producer_sha256": digest(Path(__file__)),
            "selection": "bass/keyboard pitches 21–59, velocity 75",
            "instrument_split": "Sorted by family, first floor(.65*N) train, next floor(.2*N) validation, remainder reserved",
            "reserved_inference_policy": "Only after one validation winner is frozen and all 392 earlier regressions pass",
            "no_user_audio_or_scores": True,
            "no_decoder_inference": True,
        },
    )
    records = []
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            safe_member(member)
            if member.name in seen:
                raise ValueError("Duplicate NSynth archive member")
            seen.add(member.name)
            if (
                not member.isfile()
                or not member.name.startswith("nsynth-valid/audio/")
                or not member.name.endswith(".wav")
            ):
                continue
            key = PurePosixPath(member.name).stem
            if key not in selected:
                continue
            if not re.fullmatch(r"[a-z0-9_]+-\d{3}-\d{3}", key) or member.size != 128044:
                raise ValueError("Unexpected selected NSynth PCM file")
            payload = tar.extractfile(member).read()
            path = folder / (key + ".wav")
            path.write_bytes(payload)
            records.append(
                {
                    "id": key,
                    **selected[key],
                    "member": member.name,
                    "audio": str(path),
                    "audio_sha256": digest(path),
                    "bytes": len(payload),
                }
            )
    if {i["id"] for i in records} != set(selected):
        raise ValueError("Missing selected NSynth audio")
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": sorted(records, key=lambda i: i["id"]),
            "instrument_counts": {
                g: len({i["instrument_str"] for i in records if i["group"] == g})
                for g in ("train", "validation", "reserved")
            },
            "no_decoder_inference": True,
            "no_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "notes": len(records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.archive, args.output)
