"""Acquire unambiguous generated chorale commands as train-only piano derivatives."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import yaml
from audit_cocochorales_v54 import audit, members, piano_unambiguous, voice_events
from prepare_robust_training_stems import digest, preserve

VERSION = "coco-authored-piano-commands-v54-v1"
COUNTS = {"train": 36}


def commands(voices):
    result = []
    if set(voices) != {0, 1, 2, 3}:
        raise ValueError("Incomplete source voices")
    for raw in voices.values():
        validated = voice_events(raw)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        voiced = [r for r in rows if int(float(r["pitch"])) != 0]
        if len(validated) != len(voiced):
            raise ValueError("Changed expression-to-key mapping")
        for (start, end, pitch), row in zip(validated, voiced, strict=True):
            velocity = max(1, min(127, round(1 + 126 * float(row["volume"]))))
            result.append((start, end, pitch, velocity))
    result.sort()
    if not piano_unambiguous([r[:3] for r in result]):
        raise ValueError("Ambiguous multi-voice piano key-off")
    return result


def notes(raw):
    events = np.asarray(json.loads(raw), dtype=np.float64)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or np.any(events != np.floor(events))
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 1] > 30000) or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
            or np.any((events[:, 3] < 1) | (events[:, 3] > 127))
            or not piano_unambiguous(events[:, :3])):
        raise ValueError("Invalid authored piano command clock")
    return events


def acquire(root, production, output):
    witnesses = production / "training/results/cocochorales-source-assessment-v54-v1"
    source_audit = json.loads((witnesses / "symbolic-audit.json").read_text())
    screen = json.loads((witnesses / "unambiguous-piano-screen.json").read_text())
    current = audit(root)
    if (current != source_audit or screen["source_audit_sha256"] != digest(witnesses / "symbolic-audit.json")
            or screen["retained_source_clocks"] != 36 or screen["unique_transposition_patterns"] != 36
            or not screen["no_rendering_or_inference_or_fitting"] or not screen["no_user_audio_or_scores"]):
        raise ValueError("Changed pinned source assessment or symbolic screen")
    selected = {r["source_id"]: r for r in screen["per_source"]}
    if len(selected) != 36:
        raise ValueError("Incomplete fixed source selection")
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "publisher": current["publisher"], "license": "CC BY 4.0",
        "source_audit_sha256": digest(witnesses / "symbolic-audit.json"),
        "screen_sha256": digest(witnesses / "unambiguous-piano-screen.json"),
        "inputs": current["inputs"], "selected_training_ids": sorted(selected),
        "producer_sha256": digest(Path(__file__)),
        "source_reader_sha256": digest(Path(__file__).with_name("audit_cocochorales_v54.py")),
        "mapping": "Use all 36 pre-screened nonoverlapping four-voice key clocks, with no performance-based selection. Frames are authored at 4ms. Omit explicit rest rows; preserve all voiced pitches/onsets/releases. Map source volume to integer piano velocity round(1+126*volume), clipped 1..127. No source audio, pitch-corrected F0 or human timing claims. Actual future piano commands are rounded to the pinned synthesis block clock. Never merge, truncate, extend or guess simultaneous same-key releases.",
        "partition": "All selected source parents and future audio variants are train-only. Group by publisher parent midi_file, not ensemble filename. No new validation or test data, and no claim of independence from Coconet/MIDI-DDSP pretraining or human composition catalogs. Existing held-out and release requirements remain.",
        "no_user_audio_or_scores": True, "no_audio_rendered_or_inferred": True}
    preserve(output / "plan.json", plan)
    voices, metadata = {}, {}
    for identifier, voice, raw in members(root / "expression-train-1.tar.bz2", "expression"):
        if identifier in selected:
            if digest_bytes(raw) != selected[identifier]["source_voice_sha256"][str(voice)]:
                raise ValueError("Changed selected voice bytes")
            voices.setdefault(identifier, {})[voice] = raw
    for identifier, _, raw in members(root / "metadata-train-1.tar.bz2", "metadata"):
        if identifier in selected:
            metadata[identifier] = raw
    records = []
    for identifier in sorted(selected):
        keys = commands(voices[identifier])
        source = selected[identifier]
        if len(keys) != source["notes"] or max(k[1] for k in keys) * .004 != source["duration_seconds"]:
            raise ValueError("Changed full-source command count or duration")
        parent = yaml.safe_load(metadata[identifier])["midi_file"]
        folder = output / str(identifier)
        folder.mkdir()
        for voice, raw in voices[identifier].items():
            (folder / f"voice-{voice}.csv").write_bytes(raw)
        (folder / "metadata.yaml").write_bytes(metadata[identifier])
        preserve(folder / "commands.json", keys)
        records.append({"id": f"coco-v54-{identifier:06d}", "source_id": identifier,
            "group": "train", "source_group": "coco-generated/" + parent,
            "parent_midi_file": parent, "midi": str(folder / "commands.json"),
            "sha256": digest(folder / "commands.json"), "source_voice_sha256": source["source_voice_sha256"],
            "metadata_sha256": digest(folder / "metadata.yaml"), "notes": len(keys),
            "source_seconds": source["duration_seconds"]})
    if len({r["source_group"] for r in records}) != 36:
        raise ValueError("Changed or duplicated selected parent identity")
    preserve(output / "manifest.json", {"version": VERSION, "items": records,
        "plan_sha256": digest(output / "plan.json"), "no_user_audio_or_scores": True,
        "no_audio_rendered_or_inferred": True})
    (output / "ATTRIBUTION.txt").write_text("CocoChorales, Yusong Wu et al., 2022, CC BY 4.0.\n"
        "https://magenta.tensorflow.org/datasets/cocochorales\n"
        "The Chamber Ensemble Generator: Limitless High-Quality MIR Data via Generative Modeling.\n"
        "Derived piano commands: 4ms source note-expression clocks, explicit volume-to-velocity mapping;\n"
        "rest rows omitted, ambiguous piano unisons excluded. No original ensemble audio used.\n")
    print(json.dumps({"training_sources": len(records), "keys": sum(r["notes"] for r in records)}), flush=True)


def digest_bytes(raw):
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def sources(folder):
    plan, manifest = [json.loads((folder / n).read_text()) for n in ("plan.json", "manifest.json")]
    rows = manifest["items"]
    if (plan["version"] != VERSION or manifest["version"] != VERSION or len(rows) != 36
            or manifest["plan_sha256"] != digest(folder / "plan.json")
            or plan["producer_sha256"] != digest(Path(__file__))
            or plan["source_reader_sha256"] != digest(Path(__file__).with_name("audit_cocochorales_v54.py"))
            or len({r["source_group"] for r in rows}) != 36
            or sorted(r["source_id"] for r in rows) != plan["selected_training_ids"]
            or not all(d[k] for d in (plan, manifest) for k in ("no_user_audio_or_scores", "no_audio_rendered_or_inferred"))):
        raise ValueError("Changed or incomplete authored-piano source")
    for row in rows:
        work = folder / str(row["source_id"])
        if (row["group"] != "train" or row["source_group"] != "coco-generated/" + row["parent_midi_file"]
                or Path(row["midi"]) != work / "commands.json"
                or digest(work / "commands.json") != row["sha256"]
                or digest(work / "metadata.yaml") != row["metadata_sha256"]):
            raise ValueError("Changed source content, command path or parent partition")
        voices = {v: (work / f"voice-{v}.csv").read_bytes() for v in range(4)}
        if any(digest_bytes(raw) != row["source_voice_sha256"][str(v)] for v, raw in voices.items()):
            raise ValueError("Changed source expression bytes")
        expected = commands(voices)
        if (not np.array_equal(notes((work / "commands.json").read_bytes()), expected)
                or len(expected) != row["notes"]
                or max(k[1] for k in expected) * .004 != row["source_seconds"]
                or yaml.safe_load((work / "metadata.yaml").read_bytes())["midi_file"] != row["parent_midi_file"]):
            raise ValueError("Changed command clock, velocity mapping or source parent")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    acquire(**vars(parser.parse_args()))
