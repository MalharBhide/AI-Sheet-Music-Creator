"""Prepare fixed train-only CC0 Keynata piano windows after native preflight."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import prepare_left_support_v41 as support
import soundfile as sf
from coco_piano_render_v54 import HeldWorkPiano as WholeWorkPiano
from keynata_piano_clock_v55 import notes
from keynata_piano_source_v57 import WINDOWS, sources
from offline_piano_v47 import ASSETS, BLOCK, LICENSE, RATE, TAIL
from offline_piano_v47 import contracts as piano_contracts
from pdmx_piano_clock_v53 import crop_labels, synthesis_clock
from prepare_robust_training_stems import digest, preserve
from prepare_tinysol_presence_v45 import canonical_padding

VERSION = "keynata-complete-work-piano-v57-v1"
COUNTS = {"train": 72}
VARIANTS = (.45, .7, 1.)


def contracts():
    names = ("acquire_keynata_piano_v55.py", "audit_keynata_piano_v55_v2.py",
             "keynata_piano_clock_v55.py", "keynata_piano_source_v57.py", "prepare_keynata_piano_v57.py",
             "native_keynata_piano_v57.py",
             "pdmx_piano_clock_v53.py", "pdmx_piano_render_v53.py", "coco_piano_render_v54.py",
             "offline_piano_v47.py", "prepare_tinysol_presence_v45.py")
    return {**support.contracts(), **{n: digest(Path(__file__).with_name(n)) for n in names}}


def transform(events, variant):
    if variant not in range(len(VARIANTS)):
        raise ValueError("Unknown fixed piano dynamics variant")
    events = events.copy()
    events[:, :2] += 3.
    result = synthesis_clock(events, VARIANTS[variant], whole_work=True)
    return result, {"synthesis_block_frames": BLOCK, "sample_rate": RATE,
        "velocity_multiplier": VARIANTS[variant], "tempo_map_preserved": True,
        "maximum_clock_delay_seconds": float(np.max(result[:, :2] - events[:, :2])),
        "render_tempo_bpm": 120, "human_timing_labels": False,
        "decoder_tempo_hint_is_not_source_tempo": True}


def tasks(rows):
    if ({row["id"] for row in rows} != set(WINDOWS) or len(rows) != len(WINDOWS)
            or any(not np.isfinite(row["source_seconds"]) or row["source_seconds"] <= 0
                   or int((row["source_seconds"]+3)//24) != WINDOWS[row["id"]] for row in rows)):
        raise ValueError("Changed fixed generated source window population")
    return [{"id": row["id"]+f"-window-{window:02d}-variant-{variant}",
             "source_id": row["id"], "variant": variant,
             "crop_start_seconds": window*24, "bitrate": (48, 128, 192)[variant]}
            for row in rows for window in range(int((row["source_seconds"]+3)//24))
            for variant in range(3)]


def labels(authored, job):
    reference = authored[:, :3].copy()
    pitch = reference.copy()
    pitch[:, 0] = np.maximum(0, pitch[:, 0]-.05)
    pitch[:, 1] += TAIL+.1
    return crop_labels(reference, pitch, job["crop_start_seconds"], 24)


def native_receipt(path, source, jobs):
    receipt = json.loads(path.read_text())
    rows = sources(source)
    original = {row["id"]: row for row in rows}
    expected = {job["id"]: job for job in jobs}
    records = receipt["records"]
    if (receipt["version"] != VERSION or receipt["code_sha256"] != contracts()
            or receipt["piano_assets"] != piano_contracts()
            or receipt["source_manifest_sha256"] != digest(source/"acquisition-manifest.json")
            or receipt["symbolic_audit_sha256"] != digest(source/"symbolic-audit-v2.json")
            or receipt["baseline_hashes"] != support.hashes()
            or not receipt["no_model_inference"] or not receipt["no_user_audio_or_scores"]
            or len(records) != 72 or len({r["id"] for r in records}) != 72
            or {r["id"] for r in records} != set(expected)):
        raise ValueError("Changed or incomplete native preflight contract")
    for record in records:
        job = expected[record["id"]]
        row = original[job["source_id"]]
        authored, _ = transform(notes(Path(row["midi"]).read_bytes()), job["variant"])
        reference, pitch = labels(authored, job)
        if (any(record[key] != value for key, value in job.items())
                or record["duration"] != 24 or record["sample_rate"] != RATE
                or record["source_midi_sha256"] != row["midi_sha256"]
                or record["reference_sha256"] != hashlib.sha256(reference.tobytes()).hexdigest()
                or record["pitch_support_sha256"] != hashlib.sha256(pitch.tobytes()).hexdigest()
                or record["keys"] != len(reference)
                or not re.fullmatch(r"[0-9a-f]{64}", record["wave_samples_sha256"])):
            raise ValueError("Changed native crop commands or waveform witness")
    return {record["id"]: record for record in records}


def preflight(source, output):
    rows = sources(source)
    jobs = tasks(rows)
    original = {row["id"]: row for row in rows}
    code, baseline, assets = contracts(), support.hashes(), piano_contracts()
    whole, records = {}, []
    with WholeWorkPiano() as piano:
        for job in jobs:
            row = original[job["source_id"]]
            authored, _ = transform(notes(Path(row["midi"]).read_bytes()), job["variant"])
            key = (row["id"], job["variant"])
            if key not in whole:
                piano.natural_expirations.clear()
                whole[key] = piano.render(authored)[0]
            start = job["crop_start_seconds"]*RATE
            wave = whole[key][start:start+24*RATE].copy()
            if wave.shape != (24*RATE,) or not np.isfinite(wave).all():
                raise ValueError("Invalid native fixed waveform crop")
            reference, pitch = labels(authored, job)
            records.append({**job, "duration": 24, "sample_rate": RATE,
                "source_midi_sha256": row["midi_sha256"], "keys": len(reference),
                "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "pitch_support_sha256": hashlib.sha256(pitch.tobytes()).hexdigest(),
                "wave_samples_sha256": hashlib.sha256(wave.tobytes()).hexdigest()})
            print(json.dumps({"native_crop": len(records), "total": len(jobs), "id": job["id"]}), flush=True)
    if code != contracts() or baseline != support.hashes() or assets != piano_contracts():
        raise ValueError("Native preparation inputs changed")
    preserve(output, {"version": VERSION, "code_sha256": code, "baseline_hashes": baseline,
        "piano_assets": assets, "source_manifest_sha256": digest(source/"acquisition-manifest.json"),
        "symbolic_audit_sha256": digest(source/"symbolic-audit-v2.json"), "records": records,
        "no_model_inference": True, "no_user_audio_or_scores": True})
    native_receipt(output, source, jobs)


def validate(record, saved, source, job):
    authored, clock = transform(notes(Path(source["midi"]).read_bytes()), job["variant"])
    reference, pitch = labels(authored, job)
    frames = 24*RATE
    if (any(record[k] != source[k] for k in ("group", "source_group"))
            or any(record[k] != job[k] for k in job)
            or record["corpus"] != "keynata-authored-sampled-piano"
            or record["source_midi_sha256"] != source["midi_sha256"]
            or record["synthesis_clock"] != clock or record["duration"] != frames/RATE
            or record["seconds"] != record["duration"] or record["human_timing_labels"]
            or not np.array_equal(record["reference"], reference)
            or not np.array_equal(record["pitch_reference"], pitch)):
        raise ValueError("Changed authored piano reference, partition or transform")
    item = {**record, **{key: saved[key] for key in support.KEYS},
            "reference": reference, "pitch_reference": pitch}
    events = item["events"]
    if (record["candidate_notes"] != len(events) or events.shape != (len(events), 4)
            or not np.isfinite(events).all() or np.any(events[:, 0] < 0)
            or np.any(events[:, 1] <= events[:, 0]) or np.any(events[:, 1] > record["duration"])):
        raise ValueError("Invalid sampled-piano candidate clock")
    np.testing.assert_array_equal(item["eligible"], support.eligible(events, item["duration"]))
    np.testing.assert_array_equal(item["x"], support.context_features(events, item["base_x"]))
    np.testing.assert_array_equal(item["release_neighbors"], support.neighbor_features(events, item["base_x"]))
    return support.annotate(support.validate(item))


def observation(folder, record, source, job, plan_sha):
    work = folder/record["id"]
    if (record["plan_sha256"] != plan_sha or Path(record["audio"]) != work/"decoded.wav"
            or any(digest(work/name) != record[key] for name, key in (
                ("features.npz", "cache_sha256"), ("decoded.wav", "audio_sha256"),
                ("decoded-codec.wav", "raw_codec_audio_sha256"),
                ("original.wav", "rendered_audio_sha256"), ("source.mp3", "mp3_sha256")))):
        raise ValueError("Changed sampled-piano physical cache")
    plan = json.loads((folder/"plan.json").read_text())
    receipt = json.loads(Path(plan["native_preflight"]).read_text())
    before = next(r for r in receipt["records"] if r["id"] == job["id"])
    original, original_rate = sf.read(work/"original.wav", dtype="float32")
    if (original_rate != RATE or original.shape != (24*RATE,)
            or hashlib.sha256(original.tobytes()).hexdigest() != before["wave_samples_sha256"]):
        raise ValueError("Native sample values differ from frozen preflight")
    raw, rate = sf.read(work/"decoded-codec.wav", dtype="float32")
    canonical, clock = canonical_padding(raw, round(record["duration"]*RATE), record["codec_clock"])
    audio, actual_rate = sf.read(record["audio"], dtype="float32")
    if (rate != RATE or actual_rate != RATE or clock != record["codec_clock"]
            or not np.array_equal(canonical, audio)):
        raise ValueError("Changed sampled-piano MP3 clock")
    with np.load(work/"features.npz", allow_pickle=False) as saved:
        if str(saved["plan_sha256"]) != plan_sha:
            raise ValueError("Changed sampled-piano cache plan")
        return validate(record, saved, source, job)


def prepare(source, output, native_preflight):
    rows = sources(source)
    jobs = tasks(rows)
    if len(jobs) != COUNTS["train"]:
        raise ValueError("Changed fixed crop count")
    original = {row["id"]: row for row in rows}
    native_crops = native_receipt(native_preflight, source, jobs)
    plan = {"version": VERSION, "source_root": str(source),
        "native_preflight": str(native_preflight), "native_preflight_sha256": digest(native_preflight),
        "source_manifest_sha256": digest(source/"acquisition-manifest.json"),
        "baseline_hashes": support.hashes(), "code_sha256": contracts(),
        "piano_assets": piano_contracts(), "expected_counts": COUNTS, "tasks": jobs,
        "variants": VARIANTS, "sample_rate": RATE,
        "labels": "Render each complete work continuously before fixed 24-second waveform crops. No note is retriggered at a crop. Complete key commands inside a crop supply attack/release references; crossing notes supply clipped pitch support only, never invented attack/release events. No tempo transformation or transposition. Dynamics variants and 300ms authored damping; conservative support adds 100ms. Protect first/last 2.5 seconds using the unchanged eligibility policy. Not human performance, pedal, or instrument-held-out evidence.",
        "partition": "All five generated parent compositions and all 72 windows/dynamics are fitting-only. No new validation/test data. The sample instrument is shared with NRG-CP; no new timbre or human-performance generalization claim. Positive generated command-support frames are UNKNOWN, not audible presence labels. Existing independent real-piano, NRG-CP and pitch-only gates are unchanged.",
        "codec_policy": "Require zero measured lag; only proven silent final padding may be canonicalized by the pinned TinySOL codec policy. Preserve rendered, raw-decoded and canonical waveforms and MP3 hashes.",
        "no_reserved_or_consumed_audio_inference": True, "no_user_audio_or_scores": True,
        "human_timing_labels": False}
    if output.exists():
        if json.loads((output/"plan.json").read_text()) != json.loads(json.dumps(plan)) or (output/"manifest.json").exists():
            raise ValueError("Preserve completed or changed Keynata preparation")
    else:
        output.mkdir()
        preserve(output/"plan.json", plan)
        (output/"PIANO-SAMPLE-LICENSE.txt").write_bytes(LICENSE.read_bytes())
        (output/"KEYNATA-LICENSE.txt").write_bytes((source/"LICENSE.txt").read_bytes())
        (output/"ATTRIBUTION.txt").write_text("Keynata Commons generated piano commands: CC0-1.0; source license in KEYNATA-LICENSE.txt.\nPublisher: https://carf-coder.github.io/keynata-commons/\n"+
            "Derived sampled-piano audio: dynamics variants, whole-work rendering, fixed waveform crops and authored 300ms damping.\n"
            "MuseScore General piano samples: MIT / public domain / CC0; complete local notice in PIANO-SAMPLE-LICENSE.txt.\n")
    engine, captured = support._GeneralEngine("balanced"), {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    plan_sha = digest(output/"plan.json")
    whole = {}
    with WholeWorkPiano() as piano:
        for job in jobs:
            row = original[job["source_id"]]
            work = output/job["id"]
            work.mkdir(exist_ok=True)
            if (work/"record.json").exists():
                record = json.loads((work/"record.json").read_text())
                observation(output, record, row, job, plan_sha)
            else:
                authored, synthesis_clock = transform(notes(Path(row["midi"]).read_bytes()), job["variant"])
                key = (row["id"], job["variant"])
                if key not in whole:
                    piano.natural_expirations.clear()
                    whole[key] = piano.render(authored)[0]
                start = job["crop_start_seconds"]*RATE
                wave = whole[key][start:start+24*RATE].copy()
                before = native_crops[job["id"]]
                if hashlib.sha256(wave.tobytes()).hexdigest() != before["wave_samples_sha256"]:
                    raise ValueError("Native samples changed after preflight")
                if len(wave) != 24*RATE:
                    raise ValueError("Incomplete fixed waveform crop")
                reference, pitch = labels(authored, job)
                duration = 24.
                decoded, codec_clock = support.encode(wave, RATE, work, job["bitrate"])
                original_wave, original_rate = sf.read(work/"original.wav", dtype="float32")
                if original_rate != RATE or not np.array_equal(original_wave, wave):
                    raise ValueError("Codec input changed authored piano waveform")
                audio = work/"decoded.wav"
                audio.rename(work/"decoded-codec.wav")
                decoded, codec_clock = canonical_padding(decoded, len(wave), codec_clock)
                sf.write(audio, decoded, RATE, subtype="FLOAT")
                midi = engine.predict(audio, "bass", synthesis_clock["render_tempo_bpm"])
                retained = [n for part in midi.instruments for n in part.notes]
                events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in retained], float).reshape(-1, 4)
                base = support.note_features(decoded, RATE, captured["acoustic"], retained, include_context=True)
                observed = support.observe(decoded, RATE)
                np.savez_compressed(work/"features.npz", events=events, base_x=base,
                    x=support.context_features(events, base), eligible=support.eligible(events, duration),
                    frames=support.coarse_sequences(observed["cqt"], events), attack_frames=support.sequences(observed, events),
                    release_frames=support.release_sequences(observed, events), release_neighbors=support.neighbor_features(events, base),
                    periodicity=support.recurrence(support.recurrence_observe(decoded, RATE), events), plan_sha256=plan_sha)
                record = {**job, "group": row["group"], "source_group": row["source_group"],
                    "corpus": "keynata-authored-sampled-piano", "source_midi_sha256": row["midi_sha256"],
                    "audio": str(audio), "audio_sha256": digest(audio), "cache_sha256": digest(work/"features.npz"),
                    "rendered_audio_sha256": digest(work/"original.wav"), "mp3_sha256": digest(work/"source.mp3"),
                    "raw_codec_audio_sha256": digest(work/"decoded-codec.wav"), "synthesis_clock": synthesis_clock,
                    "codec_clock": codec_clock, "duration": duration, "seconds": duration,
                    "reference": reference.tolist(), "pitch_reference": pitch.tolist(), "human_timing_labels": False,
                    "candidate_notes": len(events), "plan_sha256": plan_sha}
                observation(output, record, row, job, plan_sha)
                preserve(work/"record.json", record)
            records.append(record)
            print(json.dumps({"cached": len(records), "total": len(jobs), "id": job["id"],
                              "notes": record["candidate_notes"]}), flush=True)
    if support.hashes() != plan["baseline_hashes"] or contracts() != plan["code_sha256"] or piano_contracts() != plan["piano_assets"]:
        raise ValueError("Changed Keynata preparation inputs")
    preserve(output/"manifest.json", {"version": VERSION, "items": records, "plan_sha256": plan_sha,
        "no_reserved_or_consumed_audio_inference": True, "no_user_audio_or_scores": True,
        "human_timing_labels": False})


def load(folder):
    plan, manifest = [json.loads((folder/name).read_text()) for name in ("plan.json", "manifest.json")]
    source = Path(plan["source_root"])
    rows = sources(source)
    original = {row["id"]: row for row in rows}
    jobs = {job["id"]: job for job in tasks(rows)}
    records = manifest["items"]
    native_receipt(Path(plan["native_preflight"]), source, list(jobs.values()))
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["baseline_hashes"] != support.hashes() or plan["code_sha256"] != contracts()
            or plan["piano_assets"] != piano_contracts() or plan["sample_rate"] != RATE
            or plan["source_manifest_sha256"] != digest(source/"acquisition-manifest.json")
            or plan["native_preflight_sha256"] != digest(Path(plan["native_preflight"]))
            or digest(folder/"KEYNATA-LICENSE.txt") != digest(source/"LICENSE.txt")
            or plan["variants"] != list(VARIANTS) or plan["tasks"] != list(jobs.values())
            or manifest["plan_sha256"] != digest(folder/"plan.json") or plan["expected_counts"] != COUNTS
            or dict(Counter(r["group"] for r in records)) != COUNTS
            or len(records) != len(jobs) or {r["id"] for r in records} != set(jobs)
            or digest(folder/"PIANO-SAMPLE-LICENSE.txt") != ASSETS[LICENSE]
            or any(doc["human_timing_labels"] for doc in (plan, manifest))
            or not all(doc[k] for doc in (plan, manifest) for k in
                ("no_reserved_or_consumed_audio_inference", "no_user_audio_or_scores"))):
        raise ValueError("Changed or leaking authored-piano observations")
    return [observation(folder, r, original[jobs[r["id"]]["source_id"]], jobs[r["id"]], manifest["plan_sha256"])
            for r in records]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("native_preflight", type=Path)
    prepare(**vars(parser.parse_args()))
