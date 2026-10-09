"""Observe original real-piano MusicNet training/validation excerpts through V25."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from acquire_musicnet_subset import MD5, SECONDS, SIZE
from app.services.bass_harmonic import features as context_features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe as acoustic_observe
from attack_local_features import sequences as attack_sequences
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from periodicity_features import features, observe
from prepare_robust_training_stems import digest, preserve
from release_local_features import release_sequences
from release_neighbor_features import features as neighbor_features


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/prepare_musicnet_bass.py",
        "training/acquire_musicnet_subset.py",
        "training/periodicity_features.py",
        "training/attack_local_features.py",
        "training/release_local_features.py",
        "training/release_neighbor_features.py",
    )
    return {name: digest(root / name) for name in names}


def sources(source):
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    witness = json.loads((source / "source.json").read_text())
    if (
        manifest["plan_sha256"] != digest(source / "plan.json")
        or manifest["source_sha256"] != digest(source / "source.json")
        or plan["producer_sha256"] != digest(Path(__file__).with_name("acquire_musicnet_subset.py"))
        or plan["publisher_sha256"] != digest(source / "publisher.json")
        or plan["metadata_sha256"] != digest(source / "metadata.csv")
        or witness["bytes"] != SIZE
        or witness["md5"] != MD5
        or not witness["complete_original_object_verified"]
        or witness["license"] != "CC BY 4.0"
        or not manifest["work_disjoint"]
        or not manifest["reserved_inference_not_performed"]
        or not plan["reserved_decoder_inference_not_allowed"]
        or not manifest["no_user_audio_or_scores"]
    ):
        raise ValueError("Changed or unverified original MusicNet source")
    rows = manifest["items"]
    selected = {r["id"]: r for r in plan["selected"]}
    if (
        len(rows) < 20
        or len({r["id"] for r in rows}) != len(rows)
        or set(selected)
        != {r["id"] for r in rows} | set(manifest["excluded_selected_test_only_ids"])
    ):
        raise ValueError("Incomplete predetermined MusicNet source identities")
    groups = {
        g: {tuple(r["work"]) for r in rows if r["group"] == g}
        for g in ("train", "validation", "reserved")
    }
    if (
        groups["train"] & (groups["validation"] | groups["reserved"])
        or groups["validation"] & groups["reserved"]
    ):
        raise ValueError("MusicNet work partition leakage")
    for row in rows:
        if (
            any(row[k] != value for k, value in selected[row["id"]].items())
            or row["original_member"] != "musicnet/train_data/" + row["id"] + ".wav"
            or row["original_label_member"] != "musicnet/train_labels/" + row["id"] + ".csv"
            or len(row["excerpts"]) != 3
            or row["clock"]["sample_rate"] != 44100
        ):
            raise ValueError("Changed MusicNet recording clock, split or member")
    return rows


def references(rows, rate, first, last, original_duration):
    reference, support = [], []
    for row in rows:
        start, end, pitch, instrument = [
            int(row[k]) for k in ("start_time", "end_time", "note", "instrument")
        ]
        if (
            not 0 <= start < end <= round(original_duration * rate)
            or not 21 <= pitch <= 108
            or instrument != 1
        ):
            raise ValueError("Invalid solo-piano note label or publisher sample clock")
        if end <= first or start >= last:
            continue
        # Exact attacks remain separate from clipped ongoing holds. Expanded
        # pitch support protects ambiguous alignment edges from false REMOVE labels.
        if first <= start < last:
            reference.append([(start - first) / rate, (min(end, last) - first) / rate, pitch])
        support.append(
            [
                max(0, (start - first) / rate - 0.05),
                min((last - first) / rate, (end - first) / rate + 0.1),
                pitch,
            ]
        )
    return reference, support


def prepare(source, output):
    rows = sources(source)
    usable = [r for r in rows if r["group"] in ("train", "validation")]
    counts = {
        g: sum(len(r["excerpts"]) for r in usable if r["group"] == g)
        for g in ("train", "validation")
    }
    plan = {
        "source_root": str(source),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "expected_counts": counts,
        "mp3_bitrates": [48, 128, 192],
        "duration": SECONDS,
        "label_policy": "All publisher solo-piano labels, original 44100 sample clocks. Exact attacks beginning inside excerpt; all overlapping holds protected through pitch_reference with .05s preceding/.1s following ambiguity padding. No fabricated key-release or sustain-pedal truth. Source publisher estimates 4% label error.",
        "reserved_works_untouched": True,
        "no_user_audio_or_scores": True,
    }
    if output.exists():
        if (
            json.loads((output / "plan.json").read_text()) != plan
            or (output / "manifest.json").exists()
        ):
            raise ValueError("Preserve changed or completed MusicNet observations")
    else:
        output.mkdir()
        preserve(output / "plan.json", plan)
    engine = _GeneralEngine("balanced")
    captured = {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    for row in usable:
        labels = Path(row["labels"])
        if digest(labels) != row["labels_sha256"]:
            raise ValueError("Changed original MusicNet labels")
        notes = list(csv.DictReader(labels.open()))
        for index, crop in enumerate(row["excerpts"]):
            identifier = f"musicnet-v25-{row['id']}-{index}"
            folder = output / identifier
            folder.mkdir(exist_ok=True)
            completion = folder / "record.json"
            if completion.exists():
                record = json.loads(completion.read_text())
                if (
                    record["plan_sha256"] != digest(output / "plan.json")
                    or record["cache_sha256"] != digest(folder / "features.npz")
                    or record["audio_sha256"] != digest(folder / "decoded.wav")
                ):
                    raise ValueError("Changed completed MusicNet observation")
            else:
                path = Path(crop["audio"])
                if (
                    digest(path) != crop["audio_sha256"]
                    or crop["end_sample"] - crop["start_sample"] != SECONDS * 44100
                ):
                    raise ValueError("Changed excerpt waveform clock")
                wave, rate = sf.read(path, dtype="float32")
                if (
                    rate != 44100
                    or len(wave) != SECONDS * rate
                    or not np.isfinite(wave).all()
                    or np.max(np.abs(wave), initial=0) <= 0
                ):
                    raise ValueError("Invalid original MusicNet excerpt")
                reference, pitch_reference = references(
                    notes,
                    rate,
                    crop["start_sample"],
                    crop["end_sample"],
                    row["clock"]["original_duration"],
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(path),
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        str(plan["mp3_bitrates"][index]) + "k",
                        str(folder / "encoded.mp3"),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(folder / "encoded.mp3"),
                        "-ar",
                        "22050",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_f32le",
                        str(folder / "decoded.wav"),
                    ],
                    check=True,
                )
                audio = folder / "decoded.wav"
                wave, rate = sf.read(audio, dtype="float32")
                if rate != 22050 or len(wave) != SECONDS * rate:
                    raise ValueError("Changed MusicNet MP3 physical clock")
                midi = engine.predict(audio, "bass", 120.0)
                retained = [n for p in midi.instruments for n in p.notes]
                events = np.asarray(
                    [[n.start, n.end, n.pitch, n.velocity] for n in retained], float
                ).reshape(-1, 4)
                base_x = note_features(
                    wave, rate, captured["acoustic"], retained, include_context=True
                )
                observed = acoustic_observe(wave, rate)
                np.savez_compressed(
                    folder / "features.npz",
                    frames=coarse_sequences(observed["cqt"], events),
                    attack_frames=attack_sequences(observed, events),
                    release_frames=release_sequences(observed, events),
                    release_neighbors=neighbor_features(events, base_x),
                    events=events,
                    base_x=base_x,
                    x=context_features(events, base_x),
                    periodicity=features(observe(wave, rate), events),
                    eligible=eligible(events, float(SECONDS)),
                    plan_sha256=digest(output / "plan.json"),
                )
                record = {
                    "id": identifier,
                    "source_group": "musicnet/" + "/".join(row["work"]),
                    "publisher_id": row["id"],
                    "group": row["group"],
                    "corpus": "musicnet-real-piano",
                    "audio": str(audio),
                    "audio_sha256": digest(audio),
                    "cache_sha256": digest(folder / "features.npz"),
                    "source_audio_sha256": crop["audio_sha256"],
                    "labels_sha256": row["labels_sha256"],
                    "source_excerpt_index": index,
                    "duration": float(SECONDS),
                    "seconds": float(SECONDS),
                    "reference": reference,
                    "pitch_reference": pitch_reference,
                    "plan_sha256": digest(output / "plan.json"),
                    "candidate_notes": len(events),
                    "kind": "real-piano",
                    "bitrate": plan["mp3_bitrates"][index],
                }
                preserve(completion, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "complete": len(records),
                        "expected": sum(counts.values()),
                        "id": identifier,
                        "notes": record["candidate_notes"],
                    }
                ),
                flush=True,
            )
    if (
        counts != {g: sum(r["group"] == g for r in records) for g in counts}
        or plan["baseline_hashes"] != hashes()
    ):
        raise ValueError("Incomplete observations or changed production baseline")
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "reserved_works_untouched": True,
            "no_user_audio_or_scores": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
