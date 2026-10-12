"""Prepare progression-disjoint sampled piano with authored key clocks only."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import prepare_left_support_v41 as support
import soundfile as sf
from acquire_nrgcp_v47 import COUNTS as SOURCE_COUNTS
from acquire_nrgcp_v47 import notes, sources
from offline_piano_v47 import ASSETS, BLOCK, LICENSE, RATE, TAIL, Piano, key_clock
from offline_piano_v47 import contracts as piano_contracts
from prepare_robust_training_stems import digest, preserve
from prepare_tinysol_presence_v45 import canonical_padding

VERSION = "nrgcp-authored-sampled-piano-v47-v1"
COUNTS = {k: 3*v for k, v in SOURCE_COUNTS.items()}
VARIANTS = ((90, .45, -12), (120, .7, 0), (150, 1., 12))


def contracts():
    names = ("acquire_nrgcp_v47.py", "offline_piano_v47.py", "prepare_nrgcp_v47.py",
             "prepare_tinysol_presence_v45.py")
    return {**support.contracts(), **{n: digest(Path(__file__).with_name(n)) for n in names}}


def transform(events, variant):
    if variant not in range(len(VARIANTS)):
        raise ValueError("Unknown predeclared piano variant")
    bpm, velocity, desired = VARIANTS[variant]
    pitches = np.asarray(events)[:, 2]
    valid = [s for s in (-12, 0, 12, 24) if pitches.min()+s >= 21 and pitches.max()+s <= 108]
    if not valid:
        raise ValueError("Cannot transpose whole source into piano range")
    shift = min(valid, key=lambda s: (abs(s-desired), s))
    return key_clock(events, bpm, shift, velocity)


def tasks(rows):
    return [{"id": row["id"]+f"-variant-{variant}", "source_id": row["id"],
             "variant": variant, "bitrate": (48, 128, 192)[variant]}
            for row in rows for variant in range(3)]


def validate(record, saved, source, job):
    authored, clock = transform(notes(Path(source["midi"]).read_bytes()), job["variant"])
    reference = authored[:, :3]
    pitch = reference.copy()
    pitch[:, 0] = np.maximum(0, pitch[:, 0]-.05)
    pitch[:, 1] += TAIL+.1
    frames = int(np.ceil((round(authored[:, 1].max()*RATE)+round((TAIL+3)*RATE))/BLOCK))*BLOCK
    if (any(record[k] != source[k] for k in ("group", "source_group"))
            or any(record[k] != job[k] for k in ("id", "source_id", "variant", "bitrate"))
            or record["source_midi_sha256"] != source["sha256"]
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


def prepare(source, output):
    rows = sources(source)
    jobs = tasks(rows)
    original = {row["id"]: row for row in rows}
    plan = {"version": VERSION, "source_root": str(source),
        "source_manifest_sha256": digest(source/"manifest.json"),
        "baseline_hashes": support.hashes(), "code_sha256": contracts(),
        "piano_assets": piano_contracts(), "expected_counts": COUNTS, "tasks": jobs,
        "variants": VARIANTS, "sample_rate": RATE,
        "labels": "Synthetic authored key commands, rounded up onto the 64-frame offline synthesis clock. All chord keys share their actual command time. Whole-source piano-range transposition only; no dropped keys or guessed same-key overlaps. Per-pitch externally authored 300ms damping after key-off, with conservative pitch support through another 100ms. No pedal or human performance labels.",
        "partition": "All variants of a source remain in its progression-family partition. Exact transposition patterns are disjoint. The fixed sampled piano instrument is shared across partitions: this is a synthetic progression-held-out study, not instrument-held-out or real-piano evaluation. Existing independent real-piano gates remain required.",
        "codec_policy": "Require zero measured lag; only proven silent final padding may be canonicalized by the pinned TinySOL codec policy. Preserve rendered, raw-decoded and canonical waveforms and MP3 hashes.",
        "no_reserved_or_consumed_audio_inference": True, "no_user_audio_or_scores": True,
        "human_timing_labels": False}
    if output.exists():
        if json.loads((output/"plan.json").read_text()) != json.loads(json.dumps(plan)) or (output/"manifest.json").exists():
            raise ValueError("Preserve completed or changed NRG-CP preparation")
    else:
        output.mkdir()
        preserve(output/"plan.json", plan)
        (output/"PIANO-SAMPLE-LICENSE.txt").write_bytes(LICENSE.read_bytes())
        (output/"ATTRIBUTION.txt").write_text((source/"ATTRIBUTION.txt").read_text()+
            "Derived sampled-piano audio: tempo, velocity and whole-phrase pitch transforms; authored 300ms damping.\n"
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
    with Piano() as piano:
        for job in jobs:
            row = original[job["source_id"]]
            work = output/job["id"]
            work.mkdir(exist_ok=True)
            if (work/"record.json").exists():
                record = json.loads((work/"record.json").read_text())
                observation(output, record, row, job, plan_sha)
            else:
                authored, synthesis_clock = transform(notes(Path(row["midi"]).read_bytes()), job["variant"])
                wave, reference, pitch = piano.render(authored)
                duration = len(wave)/RATE
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
                    "corpus": "nrgcp-authored-sampled-piano", "source_midi_sha256": row["sha256"],
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
        raise ValueError("Changed NRG-CP preparation inputs")
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
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["baseline_hashes"] != support.hashes() or plan["code_sha256"] != contracts()
            or plan["piano_assets"] != piano_contracts() or plan["sample_rate"] != RATE
            or plan["source_manifest_sha256"] != digest(source/"manifest.json")
            or plan["variants"] != [list(v) for v in VARIANTS] or plan["tasks"] != list(jobs.values())
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
    prepare(**vars(parser.parse_args()))
