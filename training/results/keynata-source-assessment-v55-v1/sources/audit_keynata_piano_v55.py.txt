"""Audit new CC0 generated piano key commands; no audio or model inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from acquire_keynata_piano_v55 import COMMIT, piano_rows, verify
from pdmx_piano_clock_v53 import notes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(path, value):
    with path.open("x") as target:
        json.dump(value, target, indent=2)
        target.write("\n")


def metadata_matches(row, metadata):
    keys = ("track_id", "bpm", "category", "duration_seconds", "engine_commit", "form",
            "generated_date", "instrumentation", "length_tier", "meter_beats", "mode",
            "modulation_semitones", "section_length", "seed", "tonic", "versions")
    if (any(row[k] != metadata[k] for k in keys)
            or metadata["license"]["generated_content"] != "CC0-1.0"
            or metadata["screening"]["passed"] is not True):
        raise ValueError("Changed individual piano source identity or license")
    midi = [f for f in metadata["files"] if f["format"] == "midi"]
    if (len(midi) != 1 or midi[0]["name"] != row["track_id"]+".mid"
            or midi[0]["role"] != "main"):
        raise ValueError("Ambiguous individual piano command asset")
    return midi[0]


def audit(root, output):
    tree = json.loads((root/"github-tree.json").read_text())
    if tree["sha"] != COMMIT or tree["truncated"]:
        raise ValueError("Changed publisher tree")
    entries = {e["path"]: e for e in tree["tree"]}
    plan = json.loads((root/"acquisition-plan.json").read_text())
    manifest = json.loads((root/"acquisition-manifest.json").read_text())
    producer = Path(__file__).with_name("acquire_keynata_piano_v55.py")
    if (manifest["plan_sha256"] != digest(root/"acquisition-plan.json")
            or plan["producer_sha256"] != digest(producer) or plan["commit"] != COMMIT
            or plan["tree_sha256"] != digest(root/"github-tree.json")
            or plan["index_sha256"] != digest(root/"index.json")
            or plan["license_sha256"] != digest(root/"LICENSE.txt")
            or len(manifest["files"]) != 38
            or [r["path"] for r in manifest["files"]] != plan["paths"]
            or not manifest["no_audio_acquisition"] or not manifest["no_model_training_or_inference"]):
        raise ValueError("Changed symbolic source acquisition")
    for record in manifest["files"]:
        data = (root/record["path"]).read_bytes()
        if (verify(data, entries[record["path"]]) != record["sha256"]
                or record["git_blob_sha1"] != entries[record["path"]]["sha"]
                or len(data) != record["bytes"]):
            raise ValueError("Changed acquired piano source bytes")
    rows = piano_rows(json.loads((root/"index.json").read_text()))
    records = []
    for row in rows:
        folder = root/"release/tracks"/row["track_id"]
        metadata = json.loads((folder/"metadata.json").read_text())
        asset = metadata_matches(row, metadata)
        data = (folder/asset["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset["sha256"] or len(data) != asset["bytes"]:
            raise ValueError("Individual metadata does not authenticate source commands")
        record = {"id": row["track_id"], "group": "train-if-later-admitted",
            "source_group": "keynata-generated/"+row["track_id"],
            "source_midi_sha256": asset["sha256"], "metadata_sha256": digest(folder/"metadata.json"),
            "declared_duration_seconds": row["duration_seconds"], "license": "CC0-1.0",
            "human_performance": False}
        try:
            reference = notes(data)
        except ValueError as error:
            record.update(eligible_unambiguous_commands=False, exclusion_reason=str(error))
        else:
            # A publisher's form length is not a key-release clock. Never scale
            # source commands to match the rounded displayed duration.
            record.update(eligible_unambiguous_commands=True, keys=len(reference),
                last_key_release_seconds=float(reference[:, 1].max()),
                command_array_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                holds_over_four_seconds=int(((reference[:, 1]-reference[:, 0]) > 4).sum()),
                minimum_pitch=int(reference[:, 2].min()), maximum_pitch=int(reference[:, 2].max()),
                repeated_same_key_pairs=sum(int((np.diff(reference[reference[:, 2] == pitch, 0]) < 1).sum())
                    for pitch in np.unique(reference[:, 2])))
        records.append(record)
    result = {"version": "keynata-piano-symbolic-audit-v55-v1", "commit": COMMIT,
        "inspected_piano_sources": len(rows),
        "eligible_unambiguous_commands": sum(r["eligible_unambiguous_commands"] for r in records),
        "publisher": "https://carf-coder.github.io/keynata-commons/", "license": "CC0-1.0",
        "acquisition_manifest_sha256": digest(root/"acquisition-manifest.json"),
        "audit_producer_sha256": digest(Path(__file__)),
        "clock_reader_sha256": digest(Path(__file__).with_name("pdmx_piano_clock_v53.py")),
        "training_cleared": False, "native_preparation_and_partition_audit_pending": True,
        "no_audio_acquisition_or_inference": True, "no_user_audio_or_scores": True,
        "no_consumed_or_fresh_tests_read": True, "new_weights_deployed": False,
        "not_independent_human_piano_accuracy_evidence": True, "sources": records}
    preserve(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    audit(**vars(parser.parse_args()))
