"""Restore exact prior dataset artifacts, retaining their historical provenance.

This does not claim a new complete-archive verification. Every restored crop and
label must match the archived original artifact hash; no source reader is changed.
The working official transfer is neither stopped nor overwritten by this module.
"""

import hashlib
import json
import shutil
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "15078b3037afe83bdcacb9cea14232e4debffd2b"
BASE = "https://huggingface.co/datasets/DreamyWanderer/MusicNet/resolve/" + REVISION + "/data/train/"
MANIFEST_SHA = "51fc54ee4e30f56840e461e48d57ab6a2f8b39db219863a98ac0ab57242f9aa3"
ARCHIVE_SHA = "f26e6d023b7a62cd54d8f2b0533b88f549c30e3fad0aa546d1ea24aefba28fca"
BLOCK = 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def crops(stream, row):
    """Consume only the original first-75-second prefix; validate exact outputs."""
    consumed = 0

    def read(amount):
        nonlocal consumed
        pieces = []
        while amount:
            piece = stream.read(min(amount, BLOCK))
            if not piece or len(piece) > min(amount, BLOCK):
                raise ValueError("Truncated or unbounded original waveform read")
            amount -= len(piece)
            consumed += len(piece)
            pieces.append(piece)
        return b"".join(pieces)

    def discard(amount):
        while amount:
            size = min(amount, BLOCK)
            read(size)
            amount -= size

    header = read(12)
    if header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise ValueError("Unsupported original waveform container")
    total = struct.unpack("<I", header[4:8])[0] + 8
    position, fmt = 12, None
    while position < total:
        if position > BLOCK:
            raise ValueError("Oversized original waveform header")
        tag, length = struct.unpack("<4sI", read(8))
        position += 8
        if position + length + length % 2 > total:
            raise ValueError("Original waveform chunk escapes container")
        if tag == b"fmt ":
            if fmt is not None or not 16 <= length <= 64:
                raise ValueError("Invalid original waveform format")
            fmt = read(length)
            code, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            clock = row["clock"]
            if ((code, channels, rate, bits) != tuple(clock[k] for k in
                    ("format_code", "channels", "sample_rate", "bits"))
                    or code not in (1, 3) or channels not in (1, 2) or rate != 44100
                    or bits not in (16, 24, 32) or (code == 3 and bits != 32)
                    or align != channels * (bits // 8) or byte_rate != rate * align):
                raise ValueError("Changed original waveform clock")
        elif tag == b"data":
            if fmt is None or length % align or length / byte_rate != row["clock"]["original_duration"]:
                raise ValueError("Changed original waveform duration")
            if ([c["start_sample"] for c in row["excerpts"]] != [s * rate for s in (15, 35, 55)]
                    or [c["end_sample"] for c in row["excerpts"]] != [s * rate for s in (35, 55, 75)]):
                raise ValueError("Changed original excerpt clock or coverage")
            outputs, offset = [], 0
            for crop in row["excerpts"]:
                first = crop["start_sample"] * align
                amount = (crop["end_sample"] - crop["start_sample"]) * align
                if first < offset or amount != 20 * byte_rate or first + amount > length:
                    raise ValueError("Changed original excerpt clock")
                discard(first - offset)
                raw = read(amount)
                offset = first + amount
                body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
                wave = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
                if digest(wave) != crop["audio_sha256"]:
                    raise ValueError("Restored crop differs from original artifact")
                outputs.append(wave)
            if len(outputs) != 3:
                raise ValueError("Incomplete original excerpt coverage")
            return outputs, consumed
        else:
            if length > BLOCK:
                raise ValueError("Oversized original waveform metadata")
            discard(length)
        if length % 2:
            discard(1)
        position += length + length % 2
    raise ValueError("Missing original waveform samples")


def restore(production, output):
    from approved_model_finetune import storage_preflight

    production, output = Path(production), Path(output)
    if output.exists():
        raise ValueError("Preserve existing restored or partial artifacts")
    evidence = production / "training/results/musicnet-original-piano-source-v1"
    if digest((evidence / "manifest.json").read_bytes()) != MANIFEST_SHA:
        raise ValueError("Changed original source manifest")
    manifest = json.loads((evidence / "manifest.json").read_text())
    source_bytes = (evidence / "source.json").read_bytes()
    source = json.loads(source_bytes)
    plan_bytes = (evidence / "plan.json").read_bytes()
    original_plan = json.loads(plan_bytes)
    if (digest(source_bytes) != manifest["source_sha256"]
            or digest(plan_bytes) != manifest["plan_sha256"]
            or digest((evidence / "publisher.json").read_bytes()) != original_plan["publisher_sha256"]
            or digest((evidence / "metadata.csv").read_bytes()) != original_plan["metadata_sha256"]
            or digest((production / "training/acquire_musicnet_subset.py").read_bytes()) != original_plan["producer_sha256"]):
        raise ValueError("Changed historical source-evidence bindings")
    if (source["sha256"] != ARCHIVE_SHA or not source["complete_original_object_verified"]
            or source["bytes"] != 11097394998 or source["md5"] != "844764911fa0d5b97c97da944a057590"
            or source["license"] != "CC BY 4.0" or len(manifest["items"]) != 32):
        raise ValueError("Missing historical full-source verification and rights")
    reservation = storage_preflight(output.parent, 5_000_000_000,
        audited_links={"left-precision-v43-replay/training/results": production / "training/results"})
    output.mkdir()
    (output / "audio").mkdir()
    (output / "labels").mkdir()
    plan = {"original_manifest_sha256": MANIFEST_SHA, "original_archive_sha256": ARCHIVE_SHA,
            "immutable_mirror_revision": REVISION, "producer_sha256": digest(Path(__file__).read_bytes()),
            "storage_preflight": reservation, "new_full_archive_verification": False,
            "historical_original_verification_reused": True, "source_records": 32,
            "source_crops": 96, "source_labels": 32, "no_model_inference": True}
    (output / "restoration-plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def recover(row):
        identifier = row["id"]
        if not identifier.isdigit() or row["original_member"] != "musicnet/train_data/" + identifier + ".wav":
            raise ValueError("Changed original source identity")
        base = BASE + identifier
        with urllib.request.urlopen(base + ".csv", timeout=60) as response:
            labels = response.read(2 * BLOCK + 1)
        if len(labels) > 2 * BLOCK or digest(labels) != row["labels_sha256"]:
            raise ValueError("Restored labels differ from original artifact")
        with urllib.request.urlopen(base + ".wav", timeout=60) as response:
            waves, count = crops(response, row)
        (output / "labels" / (identifier + ".csv")).write_bytes(labels)
        for index, wave in enumerate(waves):
            (output / "audio" / (identifier + "-" + str(index) + ".wav")).write_bytes(wave)
        return {"id": identifier, "group": row["group"], "crops_verified": len(waves),
                "labels_verified": True, "waveform_prefix_bytes_read": count, "url": base}

    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for record in executor.map(recover, manifest["items"]):
            rows.append(record)
            print(json.dumps({"source_records_restored": len(rows), "total": 32, "id": record["id"]}), flush=True)
    # Copy the original provenance as historical evidence, not a new archive
    # verification. Its hashes and the native source-reader guards stay intact.
    for path in evidence.iterdir():
        if path.suffix in (".json", ".csv"):
            shutil.copy2(path, output / path.name)
    for row in manifest["items"]:
        for index, crop in enumerate(row["excerpts"]):
            if digest((output / "audio" / (row["id"] + "-" + str(index) + ".wav")).read_bytes()) != crop["audio_sha256"]:
                raise ValueError("Restored waveform changed before completion")
        if digest((output / "labels" / (row["id"] + ".csv")).read_bytes()) != row["labels_sha256"]:
            raise ValueError("Restored labels changed before completion")
    result = {**plan, "all_96_original_crops_byte_identical": True,
              "all_32_original_labels_byte_identical": True, "all_original_partitions_unchanged": True,
              "ready_for_native_source_guard_after_activation": True,
              "per_source": rows, "no_user_audio_or_scores": True}
    (output / "restoration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    restore(Path("/src"), Path("/training/musicnet-restored-bytes-v1"))
